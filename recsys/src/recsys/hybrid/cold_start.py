"""Handles the three cold starts, which are different problems:

new user   no history. Use onboarding category picks plus trending in those
           categories. Blend in CF as soon as history passes the threshold.

new item   no interactions. Content-based only - its embedding exists from
           metadata alone. Give it a small exploration budget in the feed or it
           never gets the interactions it needs.

new user   and new item at once: fall back to popularity, and log everything.
"""
