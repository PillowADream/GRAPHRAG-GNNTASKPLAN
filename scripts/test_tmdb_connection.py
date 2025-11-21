import os
import requests
import sys

def test_connection():
    # 1. 检查 Token 是否设置
    token = os.getenv("TMDB_BEARER_TOKEN")
    if not token:
        print("❌ 错误: 环境变量 TMDB_BEARER_TOKEN 未设置。")
        print("   请先执行: $env:TMDB_BEARER_TOKEN = 'eyJh...' (PowerShell)")
        return

    print(f"✅ 已检测到 TMDB_BEARER_TOKEN (前缀: {token[:10]}...)")
    
    # 2. 打印代理配置 (如果有)
    proxies = {
        "http": os.getenv("HTTP_PROXY") or os.getenv("http_proxy"),
        "https": os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    }
    print(f"ℹ️  当前代理设置: {proxies}")

    # 3. 尝试连接 TMDB
    url = "https://api.themoviedb.org/3/authentication"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    print(f"\n🔄 正在尝试连接 {url} ...")
    
    try:
        # 设置 10 秒超时
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"📡 HTTP 状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 连接成功！认证有效。")
            print("   返回数据:", response.json())
        elif response.status_code == 401:
            print("❌ 连接成功，但认证失败。请检查 Token 是否正确。")
        else:
            print(f"⚠️ 连接成功，但返回了意外的状态码: {response.text}")
            
    except requests.exceptions.ProxyError:
        print("❌ 代理错误: 无法连接到配置的代理服务器。请检查端口和地址。")
    except requests.exceptions.SSLError:
        print("❌ SSL 错误: 证书验证失败。如果您在使用企业代理，可能需要设置 requests 的 verify=False。")
    except requests.exceptions.ConnectTimeout:
        print("❌ 连接超时: 无法在 10 秒内建立连接。")
        print("   原因可能是网络被墙或防火墙拦截。")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("   这通常意味着 DNS 解析失败或完全无法访问目标主机。")
    except Exception as e:
        print(f"❌ 未知错误: {e}")

if __name__ == "__main__":
    test_connection()