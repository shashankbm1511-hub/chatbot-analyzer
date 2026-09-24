"""Generate the synthetic demo dataset shipped in ``data/sample_conversations.csv``.

Every conversation is produced from hand-written templates plus a seeded random
generator. No real customer, company, or personal data is used. Rerunning the
script with the same seed reproduces the same file byte-for-byte.

Usage:
    python scripts/generate_synthetic_data.py            # default: 240 rows
    python scripts/generate_synthetic_data.py --rows 300 --seed 7
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_conversations.csv"

CHANNELS = ["web", "mobile_app", "messaging"]
CHANNEL_WEIGHTS = [0.5, 0.35, 0.15]

# ---------------------------------------------------------------------------
# Templates per intent.
#   openers:    first customer message
#   answers:    helpful bot replies
#   rephrase:   customer repeating / clarifying after an unhelpful reply
#   agent_fix:  what a human agent says after a handoff
#   paths:      probability weights for (resolved_fast, resolved_retry,
#               unresolved_abandon, escalated)
# ---------------------------------------------------------------------------
INTENTS: dict[str, dict] = {
    "Billing": {
        "openers": [
            "Hi, I was charged twice for my subscription this month.",
            "Why is my bill higher than usual this month?",
            "I see an extra charge on my invoice that I don't recognise.",
            "Can you explain the late fee on my latest bill?",
            "My payment went through but the invoice still shows as unpaid.",
            "I need a copy of my invoice for last month for my records.",
        ],
        "answers": [
            "I can see a duplicate charge on your account. I've raised a reversal and it should reflect in 3-5 business days.",
            "Your bill includes a one-time plan upgrade charge from the 12th. You can view the itemised invoice under Billing > Invoices.",
            "The late fee was applied because the payment was received after the due date. I've applied a one-time courtesy waiver.",
            "Your payment is confirmed. The invoice status updates within 24 hours; I've refreshed it for you now.",
        ],
        "rephrase": [
            "That's not what I asked. I was charged twice, not once.",
            "I already checked the invoices page, the extra charge is still there.",
            "Again, the amount on my bill is wrong.",
        ],
        "agent_fix": [
            "Hi, this is a support specialist. I've reviewed your billing history and issued a refund for the duplicate charge.",
            "Thanks for waiting. I've corrected the invoice and emailed you the updated copy.",
        ],
        "paths": [0.45, 0.25, 0.15, 0.15],
    },
    "Account/Login": {
        "openers": [
            "I can't log in to my account, it keeps saying invalid password.",
            "I forgot my password and the reset email never arrives.",
            "My account is locked after too many attempts.",
            "How do I change the email address on my account?",
            "The two-factor code isn't coming through to my phone.",
            "I'm not able to sign in on the mobile app.",
        ],
        "answers": [
            "I've sent a new password reset link. Please check your spam folder as well; the link is valid for 30 minutes.",
            "Your account has been unlocked. You can try signing in again now.",
            "You can update your email under Settings > Profile. You'll get a verification link at the new address.",
            "I've re-sent the verification code. It can take up to two minutes to arrive.",
        ],
        "rephrase": [
            "I still haven't received the reset email.",
            "It's still saying invalid password even after the reset.",
            "The code still isn't coming. I've tried three times already.",
        ],
        "agent_fix": [
            "Hi, I'm from the account security team. I've verified your identity and reset your login manually.",
            "I've updated the phone number on your account, so the codes will now arrive correctly.",
        ],
        "paths": [0.45, 0.25, 0.12, 0.18],
    },
    "Technical Issue": {
        "openers": [
            "The app keeps crashing when I open my dashboard.",
            "The website is showing an error 500 when I try to check out.",
            "Pages are loading really slowly for me today.",
            "The export button isn't working, nothing downloads.",
            "I'm getting a blank screen after the latest update.",
            "Notifications stopped working on my phone.",
        ],
        "answers": [
            "Please update the app to the latest version and clear the cache. That resolves this crash in most cases.",
            "We had a brief service disruption that is now fixed. Could you try again?",
            "Try logging out and back in, then retry the export. It should download as a CSV file.",
            "Please check that notifications are enabled for the app in your phone settings.",
        ],
        "rephrase": [
            "I already updated the app and cleared the cache, it still crashes.",
            "Still getting the same error. This is the third time I'm trying.",
            "Nothing has changed, it's still not working.",
        ],
        "agent_fix": [
            "Hi, I'm a technical support engineer. I've logged this as a bug with our product team and applied a workaround to your account.",
            "I've reproduced the issue and pushed a fix to your account settings. Please restart the app.",
        ],
        "paths": [0.35, 0.25, 0.2, 0.2],
    },
    "Order/Transaction": {
        "openers": [
            "Where is my order? It was supposed to arrive yesterday.",
            "My payment failed but money was deducted from my account.",
            "Can I change the delivery address for my order?",
            "I received the wrong item in my order.",
            "How do I track my shipment?",
            "My order status hasn't changed in four days.",
        ],
        "answers": [
            "Your order is out for delivery and should arrive today. You can track it under Orders > Track shipment.",
            "Failed payments are automatically reversed by the bank within 5-7 business days.",
            "I've updated the delivery address. You'll receive a confirmation email shortly.",
            "Sorry about that. I've arranged a free pickup and a replacement for the correct item.",
        ],
        "rephrase": [
            "The tracking page just says 'in transit' and nothing else.",
            "It's been more than 7 days and the money still hasn't come back.",
            "I asked about my order, not a generic tracking link.",
        ],
        "agent_fix": [
            "Hi, I'm from the orders team. I've contacted the courier and your package will be delivered tomorrow.",
            "I've confirmed with our payments team and the reversal has been processed today.",
        ],
        "paths": [0.45, 0.25, 0.13, 0.17],
    },
    "Refund": {
        "openers": [
            "I returned my item two weeks ago and still haven't got my refund.",
            "How long does a refund take?",
            "I want a refund for a product that arrived damaged.",
            "My refund was approved but the amount is less than what I paid.",
            "Can I get a refund instead of store credit?",
        ],
        "answers": [
            "Your refund was processed on the 3rd. It usually takes 5-7 business days to reach your bank.",
            "Refunds are issued to the original payment method within 5-7 business days after we receive the return.",
            "I'm sorry the product was damaged. I've initiated a full refund; no return is needed.",
            "The difference is the original shipping fee, which is non-refundable under our policy.",
        ],
        "rephrase": [
            "It has been more than 7 business days already.",
            "I don't want store credit, I want my money back.",
            "That doesn't answer why I got less money back.",
        ],
        "agent_fix": [
            "Hi, I'm a refunds specialist. I've escalated your refund to priority processing; you'll receive it within 48 hours.",
            "I've reviewed your case and approved a refund of the shipping fee as well.",
        ],
        "paths": [0.3, 0.2, 0.2, 0.3],
    },
    "Cancellation": {
        "openers": [
            "I want to cancel my subscription.",
            "How do I close my account permanently?",
            "Please cancel my order, I don't need it anymore.",
            "I'm thinking of cancelling because the price went up.",
            "Cancel my plan before the next renewal date.",
        ],
        "answers": [
            "I've cancelled your subscription. You'll keep access until the end of the current billing period.",
            "You can close your account under Settings > Account > Close account. Would you like me to do it for you?",
            "Your order has been cancelled and any payment will be refunded to the original method.",
            "Before you go, you're eligible for a 20% discount for the next three months. Would you like to apply it?",
        ],
        "rephrase": [
            "I don't want a discount, I just want to cancel.",
            "I've asked to cancel twice now.",
            "The cancel button doesn't work on the settings page.",
        ],
        "agent_fix": [
            "Hi, I'm from the customer care team. I've cancelled your plan and confirmed no further charges will be made.",
            "I've closed your account and sent you a confirmation email.",
        ],
        "paths": [0.4, 0.2, 0.15, 0.25],
    },
    "Product Information": {
        "openers": [
            "Does the premium plan include priority support?",
            "What's the difference between the basic and pro plans?",
            "Is this product available in a larger size?",
            "Do you ship internationally?",
            "Does the app work offline?",
            "What is the warranty period on this product?",
        ],
        "answers": [
            "Yes, the premium plan includes priority support with faster response times.",
            "The pro plan adds team accounts, advanced reports and higher usage limits. You can compare plans on the pricing page.",
            "Yes, we ship to most countries. Shipping costs are shown at checkout.",
            "The product comes with a 12-month warranty covering manufacturing defects.",
        ],
        "rephrase": [
            "That's not what I asked, I wanted to know about the size.",
            "Can you be more specific about the limits?",
        ],
        "agent_fix": [
            "Hi, I'm a product specialist. Here are the full specifications you asked about.",
        ],
        "paths": [0.7, 0.15, 0.1, 0.05],
    },
    "Complaint": {
        "openers": [
            "This is the worst service I've ever had. Nobody is helping me.",
            "I'm very unhappy with how my issue has been handled.",
            "Your delivery partner was rude and left my parcel outside in the rain.",
            "I've contacted support four times about the same problem and it's still not fixed.",
            "I'm really frustrated, the product stopped working after one week.",
        ],
        "answers": [
            "I'm really sorry about your experience. I've logged a formal complaint and a team member will follow up within 24 hours.",
            "I apologise for the inconvenience. I've shared your feedback with the delivery team and added a credit to your account.",
        ],
        "rephrase": [
            "I don't want an apology, I want this fixed.",
            "This is ridiculous. I keep getting the same answer.",
            "You're not listening. This is the same problem again.",
        ],
        "agent_fix": [
            "Hi, I'm a senior support lead. I've read the full history and I'm personally handling your case from here.",
            "I'm sorry for the repeated trouble. I've arranged a replacement and a goodwill credit.",
        ],
        "paths": [0.15, 0.15, 0.25, 0.45],
    },
    "General Query": {
        "openers": [
            "What are your customer support hours?",
            "How do I contact you by email?",
            "Do you have a loyalty programme?",
            "Where can I find your privacy policy?",
            "Is there a student discount?",
        ],
        "answers": [
            "Our chat support is available 24/7 and phone support runs 9am-9pm, Monday to Saturday.",
            "You can reach us through the Contact page; we reply to emails within one business day.",
            "Yes, you earn points on every purchase. You can see your balance under Rewards.",
            "The privacy policy is linked at the bottom of every page.",
        ],
        "rephrase": [
            "I meant weekend hours specifically.",
            "Where exactly on the page? I can't find it.",
        ],
        "agent_fix": [
            "Hi, I'm from the customer care team. I've emailed you the details you asked for.",
        ],
        "paths": [0.7, 0.15, 0.12, 0.03],
    },
    "Other": {
        "openers": [
            "Hello?",
            "I have a question about something else.",
            "Can you help me with a partnership enquiry?",
            "I want to give some feedback about the new design.",
            "Is anyone there?",
        ],
        "answers": [
            "Hi! I'm the virtual assistant. How can I help you today?",
            "Thanks for the feedback, I've passed it on to our product team.",
            "For partnership enquiries, please use the Business page on our website.",
        ],
        "rephrase": [
            "You're not understanding my question.",
            "Can I talk to someone else about this?",
        ],
        "agent_fix": [
            "Hi, I'm a support team member. How can I help with your enquiry?",
        ],
        "paths": [0.45, 0.2, 0.25, 0.1],
    },
}

GREETINGS = ["", "", "", "Hi. ", "Hello, ", "Hey, "]
BOT_CLARIFY = [
    "I can help with that. Could you share a few more details?",
    "Sorry, I didn't quite get that. Could you rephrase your question?",
    "Here is an article that might help: Help Centre > Getting started.",
    "I understand. Let me check that for you.",
]
CUSTOMER_THANKS = [
    "Great, that worked. Thanks!",
    "Perfect, thank you so much.",
    "Okay, got it. Thanks for the help.",
    "That solved it, thanks.",
    "Thanks, that's helpful.",
]
BOT_CLOSE = [
    "Glad I could help! Is there anything else I can do for you?",
    "You're welcome! Have a great day.",
    "Happy to help. Your issue has been marked as resolved.",
]
CUSTOMER_ABANDON = [
    "This isn't helping. Forget it.",
    "Useless. I'll try again later.",
    "Still not working. I give up.",
    "That doesn't answer my question at all.",
    "Never mind.",
]
HUMAN_REQUESTS = [
    "I want to talk to a human agent.",
    "Can I speak to a real person please?",
    "Please connect me to a live agent.",
    "Transfer me to customer service, this bot isn't helping.",
    "I need to speak with a human, this is urgent.",
]
BOT_HANDOFF = "I'm transferring you to a human agent now. Please hold on."
BOT_NO_AGENT = "Sorry, all our agents are busy right now. Please try again later."
BOT_TICKET = (
    "I've created a ticket and escalated this to our specialist team. "
    "They will contact you by email within 24 hours."
)
CUSTOMER_AFTER_AGENT_OK = ["Thank you, that's sorted then.", "Finally, thanks.", "Okay, thank you for fixing it."]
CUSTOMER_AFTER_AGENT_BAD = [
    "I'll wait for the follow-up, but I'm not happy.",
    "Okay. I hope it actually gets fixed this time.",
]

PATHS = ["resolved_fast", "resolved_retry", "unresolved_abandon", "escalated"]


def _build_conversation(intent: str, rng: random.Random) -> tuple[list[str], bool, bool]:
    """Return (turns, resolved, escalated) for one synthetic conversation."""
    spec = INTENTS[intent]
    path = rng.choices(PATHS, weights=spec["paths"])[0]
    opener = rng.choice(GREETINGS) + rng.choice(spec["openers"])
    turns = [f"Customer: {opener}"]

    if path == "resolved_fast":
        turns += [
            f"Bot: {rng.choice(spec['answers'])}",
            f"Customer: {rng.choice(CUSTOMER_THANKS)}",
            f"Bot: {rng.choice(BOT_CLOSE)}",
        ]
        return turns, True, False

    if path == "resolved_retry":
        turns += [
            f"Bot: {rng.choice(BOT_CLARIFY)}",
            f"Customer: {rng.choice(spec['rephrase'])}",
            f"Bot: {rng.choice(spec['answers'])}",
            f"Customer: {rng.choice(CUSTOMER_THANKS)}",
            f"Bot: {rng.choice(BOT_CLOSE)}",
        ]
        return turns, True, False

    if path == "unresolved_abandon":
        turns += [
            f"Bot: {rng.choice(BOT_CLARIFY)}",
            f"Customer: {rng.choice(spec['rephrase'])}",
            f"Bot: {rng.choice(BOT_CLARIFY)}",
        ]
        if rng.random() < 0.4:  # some customers repeat themselves once more
            turns += [f"Customer: {opener}", f"Bot: {rng.choice(BOT_CLARIFY)}"]
        if rng.random() < 0.25:  # asked for a human, but none was available
            turns += [f"Customer: {rng.choice(HUMAN_REQUESTS)}", f"Bot: {BOT_NO_AGENT}"]
        turns.append(f"Customer: {rng.choice(CUSTOMER_ABANDON)}")
        return turns, False, False

    if rng.random() < 0.3:
        # Bot-initiated escalation: a ticket is raised for a specialist team, but
        # there is no live human in this conversation, so it ends unresolved.
        turns += [
            f"Bot: {rng.choice(BOT_CLARIFY)}",
            f"Customer: {rng.choice(spec['rephrase'])}",
            f"Bot: {BOT_TICKET}",
            f"Customer: {rng.choice(CUSTOMER_AFTER_AGENT_BAD)}",
        ]
        return turns, False, True

    # escalated with a live handoff to a human agent
    turns += [
        f"Bot: {rng.choice(BOT_CLARIFY)}",
        f"Customer: {rng.choice(spec['rephrase'])}",
        f"Bot: {rng.choice(spec['answers'])}",
        f"Customer: {rng.choice(HUMAN_REQUESTS)}",
        f"Bot: {BOT_HANDOFF}",
        f"Agent: {rng.choice(spec['agent_fix'])}",
    ]
    resolved = rng.random() < 0.6
    turns.append(
        f"Customer: {rng.choice(CUSTOMER_AFTER_AGENT_OK if resolved else CUSTOMER_AFTER_AGENT_BAD)}"
    )
    return turns, resolved, True


def generate(rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic conversation dataset."""
    rng = random.Random(seed)
    intent_names = list(INTENTS)
    # Skew the mix so some intents are more common, as in real support queues.
    intent_weights = [14, 12, 13, 15, 10, 8, 10, 7, 7, 4]
    start = datetime(2026, 4, 1)
    customer_pool = [f"CUST-{n:04d}" for n in rng.sample(range(1000, 9999), k=max(rows * 2 // 3, 1))]

    records = []
    for i in range(rows):
        intent = rng.choices(intent_names, weights=intent_weights)[0]
        turns, resolved, escalated = _build_conversation(intent, rng)
        timestamp = start + timedelta(
            days=rng.randint(0, 90), hours=rng.randint(7, 22), minutes=rng.randint(0, 59)
        )
        base_rt = rng.uniform(1.5, 6.0) + (rng.uniform(4, 20) if escalated else 0)
        records.append(
            {
                "conversation_id": f"CONV-{i + 1:05d}",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_id": rng.choice(customer_pool),
                "conversation_text": "\n".join(turns),
                "intent": intent,
                "channel": rng.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0],
                "resolved": resolved,
                "escalated": escalated,
                "response_time_seconds": round(base_rt, 1),
                "conversation_length": len(turns),
            }
        )

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    # Blank out a small number of optional fields so the loader's missing-value
    # handling is exercised on the demo data, as happens with real exports.
    for column in ("channel", "response_time_seconds"):
        idx = df.sample(frac=0.02, random_state=seed).index
        df.loc[idx, column] = None
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    df = generate(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} synthetic conversations to {args.output}")


if __name__ == "__main__":
    main()
