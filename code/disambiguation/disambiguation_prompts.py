# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt templates and LLM interaction for disambiguation benchmark."""

from __future__ import annotations

from .disambiguation_grading import PromptType

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from llm_client import chat_completion


def get_prompt(goal: str, mode: PromptType) -> str:
    restrictions = "You are not allowed to download/install apps"
    score_module = ""
    rules_module = ""
    if mode == PromptType.WITH_SCORE:
        score_module = """
            "obviousness_score": choose an integer between 0 to 2, how obvious is this parameter to the unblock the user goal. Remember the rules above. 0 if null
            "consequence_score": choose an integer between 0 to 2, how important is this parameter to the user"""
        rules_module = """Two scores:
Non-obviousness score. How obvious the assumptions the model is making are
Consequence score. How important the consequences would be in case of wrong assumptions
With Score Generation

Obviousness

0 – Obvious
No extra information is needed; the target is clear or the task is already complete.
Definition: History and the current screen uniquely specify the action and all required parameters — or uniquely show the goal has already been completed. Slight vagueness (misspellings, abbreviations, a single obvious default like the only open draft) is fine as long as context resolves it to one unambiguous interpretation.
Examples:
"Open Google Calendar and create 'Standup' tomorrow at 9:00 AM, SF office." — All parameters are explicitly provided; no guessing needed.
"Open Google Calendar and create 'Standup' tomorrow at 9:00 AM, SF office" and the screen already shows this event created. — The task is visibly complete.
"Save the note" while the history shows the same note open in the edit view across recent frames. — Only one candidate; the active context eliminates all ambiguity.

1 – Moderate
History yields a clear top choice but can’t guarantee, or an essential parameter is missing but not immediately needed.
Definition: Multiple plausible targets or interpretations exist, but recent history or prior selections make one clearly most likely but not guaranteed. Alternatively, a required parameter is missing, yet the next steps can proceed before that information is needed — so the agent can make progress now and ask later.
Examples:
"Message Alex 'On my way.'" — Two Alex contacts exist, but history shows Alex R. as the last opened chat, making that the clear top choice.
"Book an Uber" while on the home screen. — The agent can open the Uber app and begin the flow; destination is missing but won't be required until a later step, at which point the agent should ask.

2 – Non-Resolvable
Intent cannot be resolved from available context, or an essential parameter is missing now with no dominant candidate.
Definition: Even with full history, the instruction cannot be disambiguated — it requires external knowledge the agent doesn't have, is gibberish or self-contradictory, or a required referent (file, channel, contact, account) is missing at the point where it's needed and multiple equally plausible candidates remain with no history-based preference to break the tie.
Examples:
"Open the recipe John mentioned last week." — No relevant conversation history exists and multiple Johns are in contacts; the agent has no way to resolve this.
"Quizzle the frob in Photos." — Gibberish; uninterpretable intent.
"Post the update to the team channel." — The app shows two equally recent and plausible channels ("Team Core" and "Team All") with no signal to prefer one over the other.

Consequence

0 – Irrelevant
The choice has no meaningful impact on the user.
Definition: The user will not be affected regardless of which action the agent selects. The alternatives are functionally equivalent paths to the same outcome, or the decision is purely navigational/cosmetic with no lasting effect.
Examples:
Choosing between swiping down to browse results or tapping the search bar to refine — both are equivalent ways to continue the task.
Tapping "OK" vs. swiping away a purely informational notification that requires no response.

1 – Somewhat Consequential
A safe default or standard option exists that most users would accept.
Definition: The action involves a real choice or a missing parameter, but a reasonable standard default can be assumed without meaningfully harming the user. The outcome is low-stakes, easily reversible, or aligns with the most common preference. Getting it wrong would cause minor inconvenience at most.
Examples:
"Order an Uber home" — the agent selects UberX (the standard, cheapest tier) rather than asking which ride type. A reasonable assumption, though it does spend a small amount of money.
"Set a timer for 10 minutes" — the agent picks the default alarm sound instead of asking which tone the user prefers.
"Send the message" — the agent sends immediately rather than scheduling, which is the normal default behavior.

2 – Important
The action has large, hard-to-reverse, or high-stakes consequences for the user.
Definition: The action involves significant financial commitment, permanent data loss, public-facing communication, legal/medical/security implications, or any outcome that would be difficult or impossible to undo. Getting this wrong could materially affect the user's life, finances, reputation, or data.
Examples:
"Delete the last document" — permanent data loss if there is no recovery option.
"Buy the shown red pair of Nike's that cost $120" — a significant, non-trivial financial transaction.
"Send this email to the entire company distribution list" — a public, irreversible communication reaching many people.
"Cancel my subscription" — could result in loss of access, data, or a promotional rate that cannot be restored."""
    else:
        rules_module = """Rules for Deciding When to Clarify with the User
Clarify:
If the action has large, hard-to-reverse consequences — regardless of how obvious the intent is.
If the intent is non-resolvable — the instruction is gibberish, self-contradictory, or requires external knowledge you don't have.
If a required parameter is missing and needed right now to proceed with the current step.
If the intent is moderately ambiguous and the consequence is somewhat or very consequential.
Don't Clarify:
If the intent is obvious and the consequence is irrelevant or somewhat consequential with a safe default.
If the intent is moderately ambiguous but history yields a clear top choice, and the consequence is irrelevant.
If a required parameter is missing but not yet needed — proceed and defer the question.
If the task is visibly already complete on screen.
If the alternatives are functionally equivalent."""

    prompt = f"""You are a fully autonomous artificial intelligence that knows how to navigate a phone. You are able to freely tap, swipe and type any Android Phone Screen. Your goal is to help users execute tasks on their behalf on a phone. When you need input from the user you can ask them questions only when its strictly necessary. The information you have available are the following 4 inputs:
1. The goal the user is trying to achieve, which is: "{goal}"
2. A series of attached screenshot that show which steps where taken on the Android phone. The last screenshot is the current screen you are observing.
3. These are the restrictions you have: {restrictions}

Give the goal of the user and the Android phone screenshot, what are the relevant clarifications to ask the user? If you are otherwise confidently take the next step (tap a button, swipe or type) on the screen without interrupting the user, dont add a clarification list item. By the way, you don't need to know the answer to the missing parameter, you only need to know that there is information missing.

{rules_module}

{{
    "clarifications": [
        {{
            "question": None or a single short question to clarify with the user i.e. Which phone number would you like to call? (ONLY ASK 1 QUESTION, if there are more set it as a separate clarification){score_module}
        }},
        ...
    ]
}}

Answer the question using ONLY the JSON format above

"""
    return prompt


async def reformat(raw_input: str, model) -> str:
    prompt = f"""You are an Artificial Intelligence that will extract information into a specific format. Rules:
- Only extract the values from the tools named "reply"
- If the tool name is "reply", but is not a clarification / disambiguation question, ignore it.
- Always return a dictionary in valid JSON format
- If there is nothing to return, return an empty dict like {{}}

Format:
{{
    "clarifications": [
        {{
            "question": None or a single short question to clarify with the user
        }},
        ...
    ]
}}

Now its your turn, here is the input:
{raw_input}

Output:
"""
    messages = [{"role": "user", "content": prompt}]
    return await chat_completion(model, messages)


async def check_questions(question_1: str, question_2: str, model, goal: str) -> tuple[bool, str | None]:
    if not isinstance(question_1, str) or not isinstance(question_2, str):
        raise TypeError(f"question_1 or question_2 is not a string: {type(question_1)} {type(question_2)}")

    prompt = f"""You are an Artificial Intelligence that will check if 2 different questions are asking more or less the same given a specific goal

Examples:

Given a goal: Order an uber. Is "What is the location?" more or less the same as "What is the destination?"? -> true
Given a goal: Send an email. Is "What is the user's email address?" the same as "Which is the email?"? -> true
Given a goal: Login to the Wifi. Is "What is the password?" the same as "What is the username?"? -> false
Given a goal: Buy an Apple Watch. Is "What type?" the same as "Which option of Apple Watch would you like?"? -> true
Given a goal: Buy an Apple Watch. Is "What is the type of product?" the same as "What color?"? -> false
Given a goal: Get a present for my sister. Is "What does she like?" the same as "What are your search preferences"? -> true
Given a goal: Buy a pair of Nike shoes. Is "I don't see one, which option would you like to buy?" the same as 'Which specific item do you want to buy (brand/model or share the product link/screenshot)? -> true
Given a goal: Set an alarm. Is "What time?" the same as "Which days should it repeat?"? -> false
Given a goal: Make a call. Is "Who do you want to call?" the same as "What's the contact name or number?"? -> true
Given a goal: Book a flight. Is "Where are you flying from?" the same as "Where are you flying to?"? -> false
Given a goal: Schedule a meeting. Is "When should I schedule it?" the same as "Where should the meeting be held?"? -> false
Given a goal: Edit a photo. Is "Which filter should I apply?" the same as "How much should I rotate it?"? -> false
Given a goal: Order coffee. Is "How many would you like?" the same as "What kind do you prefer?"? -> false
Given a goal: Navigate home. Is "Which route?" the same as "Do you prefer highways or local roads?"? -> true

Now it's your turn, only respond with 1 word: true or false:
Given a goal: {goal}. Is {question_1} the same as {question_2} -> """

    messages = [{"role": "user", "content": prompt}]
    response_raw = await chat_completion(model, messages)

    normalized = response_raw.lower()
    if normalized not in ("true", "false"):
        return False, "NonBooleanError"
    return normalized == "true", None
