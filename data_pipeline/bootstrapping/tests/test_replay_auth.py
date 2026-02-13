#!/usr/bin/env python3
"""
Quick test script to verify replay access with JWT authentication.

Usage:
    COLONIST_JWT="<your-token>" python test_replay_auth.py

Get your JWT token:
    1. Log into colonist.io in your browser
    2. Open DevTools (F12) > Application > Cookies > colonist.io
    3. Copy the value of jwt_colonist.io
"""

import asyncio
import os
import sys
import json
import base64
from datetime import datetime
from playwright.async_api import async_playwright

GAME_ID = "191035308"  # Test game (Robijs rank #1, 76 turns)


def validate_jwt(token: str) -> dict:
    """Decode and validate JWT token."""
    try:
        parts = token.split('.')
        payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = datetime.fromtimestamp(data.get('exp', 0))

        if datetime.now() > exp:
            print(f"ERROR: JWT token expired on {exp}")
            sys.exit(1)

        print(f"JWT valid for user: {data.get('username')} (expires: {exp})")
        return data
    except Exception as e:
        print(f"Warning: Could not validate JWT: {e}")
        return {}


async def test_replay_access(jwt_token: str):
    """Test if we can access replays with the given JWT."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        )

        # Add JWT cookie
        await context.add_cookies([{
            'name': 'jwt_colonist.io',
            'value': jwt_token,
            'domain': 'colonist.io',
            'path': '/',
        }])

        page = await context.new_page()

        # Track WebSocket messages
        ws_messages = []

        cdp = await context.new_cdp_session(page)
        await cdp.send("Network.enable")

        def on_ws_received(event):
            ws_messages.append(('received', event))

        cdp.on("Network.webSocketFrameReceived", on_ws_received)

        print(f"\nLoading replay: https://colonist.io/replay?gameId={GAME_ID}&playerColor=2")
        await page.goto(f'https://colonist.io/replay?gameId={GAME_ID}&playerColor=2', wait_until='networkidle')
        await asyncio.sleep(5)

        # Check page content
        body_text = await page.inner_text('body')

        if 'Replay Error' in body_text or 'does not allow' in body_text:
            print("\nERROR: Replay access denied")
            print("Your membership may not include replay access.")
            print(f"Page text: {body_text[:300]}...")
            await page.screenshot(path='replay_denied.png')
            print("Screenshot saved to replay_denied.png")
            await browser.close()
            return False

        # Check for game canvas
        canvas = await page.query_selector('#game-canvas')
        if canvas:
            print("SUCCESS: Game canvas loaded!")
            print(f"WebSocket messages captured: {len(ws_messages)}")

            # Look for replay controls
            buttons = await page.query_selector_all('button')
            print(f"Found {len(buttons)} buttons on page")

            # Take screenshot of working replay
            await page.screenshot(path='replay_working.png')
            print("Screenshot saved to replay_working.png")

            # Try clicking next step a few times to capture game state
            print("\nAttempting to capture game states...")
            import msgpack

            game_states = []
            for i in range(5):
                # Try keyboard arrow for next step
                await page.keyboard.press('ArrowRight')
                await asyncio.sleep(0.5)

            print(f"Total WS messages after stepping: {len(ws_messages)}")

            # Try to decode messages
            decoded_count = 0
            for direction, event in ws_messages:
                try:
                    frame = event.get('response', {})
                    payload = frame.get('payloadData', '')
                    if payload:
                        raw_bytes = base64.b64decode(payload)
                        decoded = msgpack.unpackb(raw_bytes, raw=False)
                        decoded_count += 1
                        if decoded_count <= 3:
                            print(f"  Sample decoded message: {str(decoded)[:200]}...")
                except Exception:
                    pass

            print(f"Successfully decoded {decoded_count} MessagePack messages")

            await browser.close()
            return True
        else:
            print("FAILED: Game canvas not found")
            print(f"Page text: {body_text[:500]}...")
            await page.screenshot(path='replay_failed.png')
            print("Screenshot saved to replay_failed.png")
            await browser.close()
            return False


def main():
    jwt_token = os.environ.get('COLONIST_JWT')

    if not jwt_token:
        print("ERROR: No JWT token provided")
        print("\nUsage:")
        print('  COLONIST_JWT="<your-token>" python test_replay_auth.py')
        print("\nGet your JWT token:")
        print("  1. Log into colonist.io in your browser")
        print("  2. Open DevTools (F12) > Application > Cookies > colonist.io")
        print("  3. Copy the value of jwt_colonist.io")
        sys.exit(1)

    # Validate JWT
    validate_jwt(jwt_token)

    # Test replay access
    success = asyncio.run(test_replay_access(jwt_token))

    if success:
        print("\n" + "="*50)
        print("Replay access confirmed! Ready to scrape.")
        print("Run: COLONIST_JWT=\"$COLONIST_JWT\" python scrape_top_players.py --mode test-replay --game-id 191035308")
    else:
        print("\n" + "="*50)
        print("Replay access failed. Check your membership status.")


if __name__ == "__main__":
    main()
