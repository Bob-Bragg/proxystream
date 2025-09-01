"""
Enhanced DNS Resolution Module
Supports multiple DNS resolvers and record types
"""

import asyncio
import struct
import random
import re
from typing import Optional, List, Tuple

# DNS resolver configurations
DNS_PLANS = [
    # OpenDNS (A record)
    ("208.67.222.222", 53, "myip.opendns.com", 1, 1),
    ("208.67.220.220", 53, "myip.opendns.com", 1, 1),
    # Cloudflare (TXT record, CHAOS class)
    ("1.1.1.1", 53, "whoami.cloudflare", 16, 3),
    # Google (fallback)
    ("8.8.8.8", 53, "myip.opendns.com", 1, 1),
]

def _read_name(msg: bytes, i: int) -> int:
    """Read DNS name from message"""
    while True:
        if i >= len(msg):
            return i
        l = msg[i]
        if l == 0:
            return i + 1
        if l & 0xC0 == 0xC0:  # Compression pointer
            return i + 2
        i += 1 + l

def _build_dns_query(name: str, qtype: int, qclass: int) -> Tuple[bytes, int]:
    """Build DNS query packet with TCP framing"""
    tid = random.randint(0, 0xFFFF)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)  # RD=1
    qname = b"".join(struct.pack("B", len(p)) + p.encode() for p in name.split(".")) + b"\x00"
    question = struct.pack("!HH", qtype, qclass)
    payload = header + qname + question
    return struct.pack("!H", len(payload)) + payload, tid

def _parse_first_a_or_txt(resp: bytes, expect_tid: int) -> Optional[str]:
    """Parse DNS response for A or TXT records"""
    if len(resp) < 4:
        return None
    resp = resp[2:]  # Strip TCP length prefix
    if len(resp) < 12:
        return None
    
    tid, flags, qd, an, ns, ar = struct.unpack("!HHHHHH", resp[:12])
    if tid != expect_tid or an == 0:
        return None
    
    i = 12
    # Skip questions
    for _ in range(qd):
        i = _read_name(resp, i)
        i += 4  # Skip QTYPE and QCLASS
    
    # Parse first answer
    i = _read_name(resp, i)
    if i + 10 > len(resp):
        return None
    
    rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", resp[i:i+10])
    i += 10
    
    if i + rdlen > len(resp):
        return None
    
    rdata = resp[i:i+rdlen]
    
    # A record
    if rtype == 1 and rdlen == 4:
        return ".".join(str(b) for b in rdata)
    
    # TXT record
    if rtype == 16 and rdlen >= 1:
        txt_len = rdata[0]
        if txt_len <= len(rdata) - 1:
            txt_data = rdata[1:1+txt_len].decode(errors="ignore")
            # Extract IP from TXT (Cloudflare format)
            m = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", txt_data)
            return m.group(1) if m else None
    
    return None

async def _dns_query_tcp(
    host: str,
    port: int,
    name: str,
    qtype: int,
    qclass: int,
    timeout: float = 5.0
) -> Optional[str]:
    """Perform DNS query over TCP"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout
        )
        
        q, tid = _build_dns_query(name, qtype, qclass)
        writer.write(q)
        await writer.drain()
        
        # Read response length
        ln = await asyncio.wait_for(reader.readexactly(2), timeout)
        (n,) = struct.unpack("!H", ln)
        
        # Read response payload
        payload = await asyncio.wait_for(reader.readexactly(n), timeout)
        
        writer.close()
        try:
            await writer.wait_closed()
        except:
            pass
        
        return _parse_first_a_or_txt(ln + payload, tid)
    except Exception:
        return None

async def get_exit_ip_dns_direct() -> Optional[str]:
    """Get client IP via DNS without proxy"""
    for host, port, name, qtype, qclass in DNS_PLANS:
        ip = await _dns_query_tcp(host, port, name, qtype, qclass)
        if ip:
            return ip
    return None

async def get_exit_ip_dns_via_chain(chain: List[Any]) -> Optional[str]:
    """Get exit IP using DNS over TCP through HTTP/HTTPS chain"""
    # Check chain compatibility
    if any(p.protocol not in ("http", "https") for p in chain):
        return None  # DNS over TCP requires HTTP/HTTPS proxies only
    
    import base64
    
    async def connect_chain_http(dest_host: str, dest_port: int):
        """Build TCP tunnel through HTTP CONNECT"""
        reader = None
        writer = None
        
        try:
            # Connect to first proxy
            r, w = await asyncio.open_connection(chain[0].host, chain[0].port)
            reader, writer = r, w
            
            for i, hop in enumerate(chain):
                # Determine next destination
                if i == len(chain) - 1:
                    next_host, next_port = dest_host, dest_port
                else:
                    next_host, next_port = chain[i + 1].host, chain[i + 1].port
                
                # Build CONNECT request
                req = f"CONNECT {next_host}:{next_port} HTTP/1.1\r\n"
                req += f"Host: {next_host}:{next_port}\r\n"
                req += "Connection: keep-alive\r\n"
                
                # Add auth if needed
                if hop.username and hop.password:
                    token = base64.b64encode(
                        f"{hop.username}:{hop.password}".encode()
                    ).decode()
                    req += f"Proxy-Authorization: Basic {token}\r\n"
                req += "\r\n"
                
                writer.write(req.encode("ascii"))
                await writer.drain()
                
                # Read status line
                status_line = await reader.readuntil(b"\r\n")
                if b"200" not in status_line:
                    raise RuntimeError(f"CONNECT failed: {status_line}")
                
                # Drain headers
                while True:
                    line = await reader.readuntil(b"\r\n")
                    if line in (b"\r\n", b"\n", b""):
                        break
            
            return reader, writer
            
        except Exception as e:
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except:
                    pass
            raise
    
    # Try DNS queries through chain
    for host, port, name, qtype, qclass in DNS_PLANS[:2]:  # Only OpenDNS for chains
        try:
            rd, wr = await connect_chain_http(host, port)
            try:
                query, tid = _build_dns_query(name, qtype, qclass)
                wr.write(query)
                await wr.drain()
                
                len_bytes = await rd.readexactly(2)
                resp_len = struct.unpack("!H", len_bytes)[0]
                payload = await rd.readexactly(resp_len)
                
                ip = _parse_first_a_or_txt(len_bytes + payload, tid)
                if ip:
                    return ip
            finally:
                wr.close()
                try:
                    await wr.wait_closed()
                except:
                    pass
        except:
            continue
    
    return None
