"""
ProxyStream Modules
Enhanced proxy validation and management components
"""

from .validator import AsyncQueueValidator, QueueBasedValidator, ValidationResult
from .dns_resolver import get_exit_ip_dns_direct, get_exit_ip_dns_via_chain, DNS_PLANS
from .proxy_parser import ProxyParser, parse_proxy_line
from .chain_tester import ChainTester, test_chain_comprehensive

__version__ = "1.0.0"
__all__ = [
    'AsyncQueueValidator',
    'QueueBasedValidator',
    'ValidationResult',
    'get_exit_ip_dns_direct',
    'get_exit_ip_dns_via_chain',
    'DNS_PLANS',
    'ProxyParser',
    'parse_proxy_line',
    'ChainTester',
    'test_chain_comprehensive'
]
