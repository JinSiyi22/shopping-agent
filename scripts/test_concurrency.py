import asyncio
import time

import httpx


URL = "http://127.0.0.1:8000/chat"


async def send_request(
    client: httpx.AsyncClient,
    message: str,
):
    start = time.perf_counter()

    response = await client.post(
        URL,
        json={
            "message": message
        },
        timeout=120,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"{elapsed:.2f}s | "
        f"{response.status_code}"
    )


async def main():

    async with httpx.AsyncClient() as client:

        await asyncio.gather(
            send_request(
                client,
                "预算8000，推荐办公电脑",
            ),
            send_request(
                client,
                "预算12000，推荐编程电脑",
            ),
            send_request(
                client,
                "推荐适合游戏的电脑",
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())