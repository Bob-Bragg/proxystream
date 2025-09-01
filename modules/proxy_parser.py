"""
Proxy Parsing Module
Handles various proxy formats and sources
"""

from typing import Optional, List, Dict, Any
from urllib.parse import urlsplit
import re

class ProxyParser:
    """Advanced proxy parser with multiple format support"""
    
    # Common proxy ports by protocol
    DEFAULT_PORTS = {
        'http': 8080,
        'https': 8080,
        'socks4': 1080,
        'socks5': 1080,
    }
    
    # Port hints for protocol detection
    PORT_HINTS = {
        1080: 'socks5',
        1081: 'socks4',
        3128: 'http',
        8080: 'http',
        8888: 'http',
        9050: 'socks5',  # Tor
    }
    
    @staticmethod
    def parse_proxy_line(line: str, default_protocol: str = "http") -> Optional[Dict[str, Any]]:
        """
        Parse a single proxy line in various formats
        Returns dict with: host, port, protocol, username, password
        """
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#') or line.startswith('//'):
            return None
        
        # Try different parsing strategies
        result = (
            ProxyParser._parse_url_format(line) or
            ProxyParser._parse_ip_port_format(line, default_protocol) or
            ProxyParser._parse_auth_format(line, default_protocol)
        )
        
        return result
    
    @staticmethod
    def _parse_url_format(line: str) -> Optional[Dict[str, Any]]:
        """Parse URL format: protocol://[user:pass@]host:port"""
        if "://" not in line:
            return None
        
        try:
            u = urlsplit(line)
            if not u.hostname:
                return None
            
            # Determine port
            port = u.port
            if not port:
                port = ProxyParser.DEFAULT_PORTS.get(u.scheme, 8080)
            
            return {
                'host': u.hostname,
                'port': port,
                'protocol': u.scheme or 'http',
                'username': u.username,
                'password': u.password
            }
        except:
            return None
    
    @staticmethod
    def _parse_ip_port_format(line: str, default_protocol: str) -> Optional[Dict[str, Any]]:
        """Parse IP:PORT format"""
        # Handle IPv6
        if line.startswith('['):
            match = re.match(r'\[([^\]]+)\]:(\d+)', line)
            if match:
                return {
                    'host': match.group(1),
                    'port': int(match.group(2)),
                    'protocol': default_protocol,
                    'username': None,
                    'password': None
                }
        
        # Handle IPv4
        parts = line.split(':')
        if len(parts) == 2:
            try:
                # Validate IP
                ip_parts = parts[0].split('.')
                if len(ip_parts) == 4:
                    # Check if all parts are valid numbers
                    if all(0 <= int(p) <= 255 for p in ip_parts):
                        port = int(parts[1])
                        # Guess protocol from port
                        protocol = ProxyParser.PORT_HINTS.get(port, default_protocol)
                        return {
                            'host': parts[0],
                            'port': port,
                            'protocol': protocol,
                            'username': None,
                            'password': None
                        }
            except (ValueError, AttributeError):
                pass
        
        return None
    
    @staticmethod
    def _parse_auth_format(line: str, default_protocol: str) -> Optional[Dict[str, Any]]:
        """Parse formats with authentication"""
        # Format: host:port:user:pass
        parts = line.split(':')
        if len(parts) == 4:
            try:
                return {
                    'host': parts[0],
                    'port': int(parts[1]),
                    'protocol': default_protocol,
                    'username': parts[2],
                    'password': parts[3]
                }
            except ValueError:
                pass
        
        # Format: user:pass@host:port
        if '@' in line:
            match = re.match(r'([^:]+):([^@]+)@([^:]+):(\d+)', line)
            if match:
                return {
                    'host': match.group(3),
                    'port': int(match.group(4)),
                    'protocol': default_protocol,
                    'username': match.group(1),
                    'password': match.group(2)
                }
        
        return None
    
    @staticmethod
    def parse_proxy_list(content: str, default_protocol: str = "http") -> List[Dict[str, Any]]:
        """Parse multiple proxies from text content"""
        proxies = []
        
        for line in content.splitlines():
            proxy = ProxyParser.parse_proxy_line(line, default_protocol)
            if proxy:
                proxies.append(proxy)
        
        return proxies

def parse_proxy_line(line: str, default_protocol: str = "http") -> Optional[Dict[str, Any]]:
    """Convenience function for parsing single proxy"""
    return ProxyParser.parse_proxy_line(line, default_protocol)
