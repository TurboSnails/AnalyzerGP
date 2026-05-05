import asyncio
import httpx

params = {
    "name": "yuan",
    "age": 30
}

async def hello():
    print("start")
    await asyncio.sleep(1)
    print("end")

async def fetch(url, param):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=param)
        print(response.json())

# ✅ 唯一入口
async def main():
    await hello()
    await fetch("https://httpbin.org/get", params)

if __name__ == "__main__":
    asyncio.run(main())