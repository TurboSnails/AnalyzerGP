import sys
import asyncio
import httpx

async def hello():
    print("start")
    await asyncio.sleep(1)
    print("end")

def main():
    print("hello world")

params = {
    "name": "yuan",
    "age": 30
}


if __name__ == "__main__":
    main()
    asyncio.run(hello())
    response = httpx.get("https://httpbin.org/get", params=params)
    print(response.json())


