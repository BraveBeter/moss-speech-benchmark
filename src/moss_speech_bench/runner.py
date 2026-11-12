"""Closed-loop HTTP dispatch with a concurrency-sized connection pool."""

import asyncio
import time

import aiohttp


async def dispatch(samples, client, concurrency, timeout):
    semaphore = asyncio.Semaphore(concurrency)

    async def send(session, sample):
        async with semaphore:
            return await client.send(session, sample)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout), connector=connector
    ) as session:
        started = time.perf_counter()
        tasks = [asyncio.create_task(send(session, sample)) for sample in samples]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return results, time.perf_counter() - started
