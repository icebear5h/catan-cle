"""Player chips and the message board over a snapshot with revealed hands."""

from playwright.sync_api import Page, expect


def test_player_chips_show_exact_hand_contents(mounted_revealed_hands: Page) -> None:
    page = mounted_revealed_hands
    red = page.locator(".player-state-chip").first
    blue = page.locator(".player-state-chip").nth(1)
    hand = red.locator(".player-chip-hand")

    # Contents are part of the chip: exact resources and unplayed development
    # cards, with the public totals still rendered above them. No toggle.
    expect(red.locator(".player-chip-stats")).to_contain_text("6 Hand")
    expect(red.locator(".player-chip-stats")).to_contain_text("3 Dev")
    assert [
        card.get_attribute("title")
        for card in hand.locator(".chip-hand-card").all()
    ] == ["3 WOOD", "1 SHEEP", "2 ORE", "2 KNIGHT", "1 VICTORY POINT"]
    assert hand.locator(".chip-hand-card b").all_text_contents() == ["3", "1", "2", "2", "1"]
    assert hand.locator(".chip-hand-card.dev").all_text_contents() == ["K2", "VP1"]
    assert [
        card.get_attribute("title")
        for card in blue.locator(".chip-hand-card").all()
    ] == ["1 BRICK"]

    # A player holding nothing says so rather than dropping the row.
    expect(page.locator(".player-state-chip").nth(2).locator(".chip-hand-empty")).to_have_text("empty")

    # Nothing in the gameplay dock toggles this.
    expect(page.get_by_role("button", name="Hands", exact=True)).to_have_count(0)


def test_message_board_is_present_before_anyone_speaks(
    mounted_revealed_hands: Page,
) -> None:
    page = mounted_revealed_hands
    # This fixture's game has no speech at all: the board must still be there,
    # naming itself and saying it is empty, rather than disappearing.
    board = page.locator("details.inspector-table-talk")
    expect(board).to_have_count(1)
    expect(board.locator("summary")).to_contain_text("Messages")
    expect(board.locator("summary")).to_contain_text("0 messages")
    expect(board.locator(".table-talk-empty")).to_contain_text("No messages yet")
    assert board.locator(".table-talk-entry").count() == 0
