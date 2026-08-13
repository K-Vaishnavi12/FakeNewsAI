"""Out-of-band smoke test for the four VeriTruth MCP servers.

Spawns each server over stdio, performs the MCP handshake, lists its tools and
calls one tool for real. Not part of the pytest suite (it spawns subprocesses
and is slow); run it manually:

    python scripts/smoke_mcp.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES: list[tuple[str, str, dict]] = [
    ("classifier_server", "classify_news", {"text": "Doctors are SHOCKED by this one weird trick!!!"}),
    ("evidence_server", "search_evidence", {"claim": "vaccines cause autism", "k": 2}),
    ("factcheck_server", "search_fact_checks", {"query": "bleach cures coronavirus", "k": 2}),
    ("explainer_server", "explain_prediction", {"text": "SHOCKING secret cure exposed", "top_k": 5}),
]


async def check(module: str, tool: str, args: dict) -> bool:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", f"src.mcp_servers.{module}"],
        cwd=str(ROOT),
    )

    async def run() -> bool:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                result = await session.call_tool(tool, args)
                payload = result.content[0].text if result.content else ""
                print(f"OK   {module:<20} tools={names}", flush=True)
                print(f"     {tool}(...) -> {payload[:140]}", flush=True)
                return tool in names

    try:
        return await asyncio.wait_for(run(), timeout=180)
    except Exception as exc:
        print(f"FAIL {module:<20} {type(exc).__name__}: {exc}", flush=True)
        return False


async def main() -> int:
    print("\n--- STEP 7 VERIFICATION (MCP servers) -------------------------")
    results = [await check(*case) for case in CASES]
    ok = sum(results)
    print(f"\n{ok}/{len(CASES)} MCP servers responded to a real tool call.")
    print("Expected: 4/4, each listing its tool and returning a payload")
    print("---------------------------------------------------------------\n")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
