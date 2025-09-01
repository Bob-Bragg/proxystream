"""
Proxy Chain Testing Module
Tests multi-hop proxy chains with various methods
"""

import asyncio
import aiohttp
from aiohttp_socks import ChainProxyConnector
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChainTester:
    """Comprehensive chain testing functionality"""
    
    def __init__(self, timeout: int = 30, no_apis: bool = False):
        self.timeout = timeout
        self.no_apis = no_apis
    
    async def test_chain(self, chain: List[Any]) -> Dict[str, Any]:
        """Test proxy chain with multiple methods"""
        if not chain:
            return {"success": False, "error": "Empty chain"}
        
        result = {
            "success": False,
            "hop_count": len(chain),
            "chain_protocols": [p.protocol for p in chain],
            "tests_performed": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Test 1: DNS-based (if supported)
        if all(p.protocol in ("http", "https") for p in chain):
            from .dns_resolver import get_exit_ip_dns_via_chain
            try:
                exit_ip = await get_exit_ip_dns_via_chain(chain)
                if exit_ip:
                    result["exit_ip_dns"] = exit_ip
                    result["tests_performed"].append("DNS")
                    result["success"] = True
            except Exception as e:
                result["dns_error"] = str(e)[:100]
        
        # Test 2: HTTP-based (if not in no-API mode)
        if not self.no_apis:
            try:
                urls = [p.as_url() for p in chain]
                connector = ChainProxyConnector.from_urls(urls)
                
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session:
                    # Get exit IP
                    async with session.get("https://api.ipify.org?format=json") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            exit_ip = data.get("ip")
                            result["exit_ip_http"] = exit_ip
                            result["tests_performed"].append("HTTP")
                            result["success"] = True
                    
                    # Test anonymity
                    async with session.get("https://httpbin.org/headers") as resp:
                        if resp.status == 200:
                            headers = await resp.json()
                            headers_data = headers.get("headers", {})
                            
                            # Check for leaked information
                            anonymity = "elite"
                            if "Via" in headers_data or "X-Forwarded-For" in headers_data:
                                anonymity = "anonymous"
                            if chain[0].host in str(headers_data):
                                anonymity = "transparent"
                            
                            result["anonymity"] = anonymity
                            result["tests_performed"].append("Anonymity")
            except Exception as e:
                result["http_error"] = str(e)[:100]
        
        # Test 3: Basic connectivity
        try:
            urls = [p.as_url() for p in chain]
            connector = ChainProxyConnector.from_urls(urls)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Test with robots.txt
                async with session.get("https://www.google.com/robots.txt") as resp:
                    if resp.status in [200, 301, 302, 403, 404]:
                        result["connectivity"] = True
                        result["tests_performed"].append("Connectivity")
                        result["success"] = True
        except Exception as e:
            result["connectivity_error"] = str(e)[:100]
        
        return result
    
    async def test_chain_latency(self, chain: List[Any]) -> Optional[float]:
        """Measure chain latency"""
        import time
        
        try:
            urls = [p.as_url() for p in chain]
            connector = ChainProxyConnector.from_urls(urls)
            
            start = time.perf_counter()
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get("https://www.cloudflare.com/robots.txt") as resp:
                    if resp.status in [200, 301, 302, 403, 404]:
                        return (time.perf_counter() - start) * 1000
        except:
            return None
        
        return None

async def test_chain_comprehensive(
    chain: List[Any],
    timeout: int = 30,
    no_apis: bool = False,
    geo_reader=None
) -> Dict[str, Any]:
    """Comprehensive chain testing with geo lookup"""
    tester = ChainTester(timeout=timeout, no_apis=no_apis)
    result = await tester.test_chain(chain)
    
    # Add geo information if available
    if result.get("success") and geo_reader:
        exit_ip = result.get("exit_ip_dns") or result.get("exit_ip_http")
        if exit_ip:
            try:
                # Assuming geo_lookup function is available
                from ..streamlit_app import geo_lookup
                geo = geo_lookup(exit_ip, geo_reader)
                result["exit_geo"] = geo
            except:
                pass
    
    return result
