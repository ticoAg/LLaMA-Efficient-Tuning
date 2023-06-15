from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi
YOUR_ACCESS_TOKEN = '68c52659-a191-4ed4-a7fd-7977ad10a57a'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)
# snapshot_download('ticoAg/tigerbot-7b-base', cache_dir='.empty',revision='v1.0.0')
# snapshot_download('ticoAg/bloomz-6b4-mt-zh', cache_dir='ckpt',revision='v1.0.0')
snapshot_download('baichuan-inc/baichuan-7B', cache_dir='ckpt',revision='v1.0.0')