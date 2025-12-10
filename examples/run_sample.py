import json
import sys
from pathlib import Path
# Ensure project root is on sys.path so we can import `app` when running this
# script from the `examples/` folder directly.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app import get_prediction

# ⚠️ 注意：示例数据拷贝自原始示例，长度需与模型的 FEATURE_COLUMNS 一致
SAMPLE_RAW_DATA = [
    54865, 3, 2, 0, 12, 0, 6, 6, 6.0, 0.0, 0, 0, 0.0, 0.0, 4000000.0,
    666666.6667, 3.0, 0.0, 3, 3, 3, 3.0, 0.0, 3, 3, 0, 0.0, 0.0, 0, 0,
    0, 0, 0, 0, 40, 0, 666666.6667, 0.0, 6, 6, 6.0, 0.0, 0.0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 9.0, 6.0, 0.0, 40, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0,
    33, -1, 1, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
]

if __name__ == '__main__':
    result = get_prediction(SAMPLE_RAW_DATA)
    print("\n--- 模拟网站/API 接口返回结果 ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    if result.get('status') == 'success':
        if result.get('predicted_label', '').upper() == 'BENIGN':
            print("\n🟢 预测：流量正常 (BENIGN)")
        else:
            print(f"\n🔴 预测：检测到恶意攻击！类型为 {result.get('predicted_label')}")
            attack_types = {
                'DDOS': '分布式拒绝服务攻击',
                'DOS': '拒绝服务攻击',
                'PORTSCAN': '端口扫描攻击',
                'BOT': '僵尸网络活动',
                'INFLITRATION': '渗透攻击',
                'BRUTEFORCE': '暴力破解攻击',
                'SQLINJECTION': 'SQL注入攻击',
                'XSS': '跨站脚本攻击',
                'FTP-PATATOR': 'FTP密码爆破',
                'SSH-PATATOR': 'SSH密码爆破'
            }
            attack_description = attack_types.get(result.get('predicted_label','').upper(), '未知攻击类型')
            print(f"攻击类型描述: {attack_description}")
