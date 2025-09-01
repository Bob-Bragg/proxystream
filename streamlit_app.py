# At the top, add imports
from modules.validator import AsyncQueueValidator, QueueBasedValidator, validate_until_target
from modules.dns_resolver import get_exit_ip_dns_direct, get_exit_ip_dns_via_chain
from modules.proxy_parser import ProxyParser
from modules.chain_tester import test_chain_comprehensive

# In sidebar, add NO_APIS toggle
NO_APIS = st.sidebar.toggle("🌿 No-API Mode", value=False,
                            help="Avoid external APIs, use only DNS and robots.txt")

# Replace validate_batch_concurrent
async def validate_batch_concurrent(
    proxies: List[ProxyInfo], 
    verify_tls: bool = True,
    max_concurrent: int = 10,
    progress_callback=None
) -> List[ProxyInfo]:
    validator = AsyncQueueValidator(
        max_concurrent=max_concurrent,
        timeout=10,
        no_apis=NO_APIS
    )
    return await validator.validate_batch(
        proxies,
        progress_callback=progress_callback
    )

# Add Quick Harvest button
if st.sidebar.button("🎯 Harvest 200 Working", type="primary", use_container_width=True):
    with st.spinner("Harvesting and validating..."):
        all_proxies = []
        for category in st.session_state.selected_sources:
            for src in PROXY_SOURCES[category]:
                proxies = run_async(fetch_proxies(src, 500))
                all_proxies.extend(proxies)
        
        validated = run_async(
            validate_until_target(
                all_proxies,
                target=200,
                max_concurrent=40,
                no_apis=NO_APIS
            )
        )
        st.session_state.proxies_validated.extend(validated)
        st.session_state.proxies_validated = list({p: None for p in st.session_state.proxies_validated}.keys())
        st.success(f"✅ Found {len(validated)} working proxies!")
