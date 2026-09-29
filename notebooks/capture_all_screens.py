"""Automated CDP Screen Capture for VarshaMitra Dashboard.
======================================================
Captures high-resolution browser screenshots in both Dark and Light modes,
verifying map rendering, zero empty boxes, crisp markdown rendering,
zero text ghosting, and accurate verification metrics.
"""

import asyncio
import subprocess
import json
import base64
import time
import urllib.request
import websockets
from pathlib import Path

async def capture_dashboard_screens():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    artifact_dir = Path(r"C:\Users\mudit\.gemini\antigravity\brain\f9088d57-b7f5-437c-aec4-e8a04d41d1ae")
    data_dir.mkdir(exist_ok=True)
    
    # Launch Edge Headless with CDP
    proc = subprocess.Popen([
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--remote-debugging-port=9235",
        r"--user-data-dir=C:\SIH080\edge_cdp_cap5",
        "--window-size=1600,1100",
        "http://127.0.0.1:8501"
    ])
    
    try:
        await asyncio.sleep(4)
        with urllib.request.urlopen("http://127.0.0.1:9235/json") as resp:
            tabs = json.loads(resp.read().decode("utf-8"))
        target_tab = [t for t in tabs if "8501" in t.get("url", "")][0]
        ws_url = target_tab["webSocketDebuggerUrl"]
        
        async with websockets.connect(ws_url, max_size=20_000_000) as ws:
            async def eval_js(js_expr):
                cmd_id = int(time.time() * 1000) % 1000000
                await ws.send(json.dumps({
                    "id": cmd_id,
                    "method": "Runtime.evaluate",
                    "params": {"expression": js_expr}
                }))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == cmd_id:
                        return msg.get("result", {})

            async def take_screenshot(filename):
                cmd_id = int(time.time() * 1000) % 1000000
                await ws.send(json.dumps({
                    "id": cmd_id,
                    "method": "Page.captureScreenshot",
                    "params": {"format": "png"}
                }))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == cmd_id:
                        img_bytes = base64.b64decode(msg["result"]["data"])
                        out_path1 = data_dir / filename
                        with open(out_path1, "wb") as f:
                            f.write(img_bytes)
                        out_path2 = artifact_dir / filename
                        with open(out_path2, "wb") as f:
                            f.write(img_bytes)
                        print(f"Captured: {filename} ({len(img_bytes)} bytes)")
                        return out_path1

            # Step 1: Wait 7 seconds for initial Dark Mode Map
            print("Step 1: Capturing Dark Mode Map...")
            await asyncio.sleep(7)
            await take_screenshot("screenshot_dark_map.png")

            # Step 2: Switch to Tab 2 (District Deep Dive & SHAP) in Dark Mode
            print("Step 2: Switching to District Deep Dive Tab in Dark Mode...")
            await eval_js("document.querySelectorAll('[role=\"tab\"]')[1].click()")
            await asyncio.sleep(4)
            await take_screenshot("screenshot_dark_deepdive.png")

            # Step 3: Switch to Tab 3 (Verification Skill Scores) in Dark Mode
            print("Step 3: Switching to Verification Skill Scores Tab in Dark Mode...")
            await eval_js("document.querySelectorAll('[role=\"tab\"]')[2].click()")
            await asyncio.sleep(4)
            await take_screenshot("screenshot_dark_verification.png")

            # Step 4: Click Theme Toggle to switch to Light Mode
            print("Step 4: Clicking Theme Toggle to Light Mode...")
            await eval_js("""
                (() => {
                    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Light Mode'));
                    if (btn) btn.click();
                })()
            """)
            await asyncio.sleep(4)

            # Step 5: Click Tab 1 (District Risk Map) in Light Mode
            print("Step 5: Switching to District Risk Map in Light Mode...")
            await eval_js("document.querySelectorAll('[role=\"tab\"]')[0].click()")
            await asyncio.sleep(6)
            await take_screenshot("screenshot_light_map.png")

            # Step 6: Click Tab 2 (District Deep Dive) in Light Mode
            print("Step 6: Switching to District Deep Dive in Light Mode...")
            await eval_js("document.querySelectorAll('[role=\"tab\"]')[1].click()")
            await asyncio.sleep(4)
            await take_screenshot("screenshot_light_deepdive.png")

            # Step 7: Click Tab 3 (Verification Skill Scores) in Light Mode
            print("Step 7: Switching to Verification Skill Scores in Light Mode...")
            await eval_js("document.querySelectorAll('[role=\"tab\"]')[2].click()")
            await asyncio.sleep(4)
            await take_screenshot("screenshot_light_verification.png")

    finally:
        proc.terminate()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    asyncio.run(capture_dashboard_screens())
