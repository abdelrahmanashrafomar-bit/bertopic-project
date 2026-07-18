"""
Agreement test: Compare BERTopic Transform vs Centroid Similarity

Runs both inference methods on a set of sample complaints
and computes the agreement rate.

Usage:
    python test_agreement.py

No external data files needed — complaints are embedded in the script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.inference.predict import predict_with_centroids, load_lookup
from src.config import load_config, get_project_root
from src.validators import ensure_file_exists


# ── Sample complaints (taken from the original CFPB dataset) ──
TEST_COMPLAINTS = [
    # Identity theft / fraud
    "Someone stole my identity and opened accounts in my name without my permission",
    "I am a victim of identity theft and need these fraudulent accounts removed from my credit report",
    "An unauthorized person used my personal information to apply for credit",
    "Someone opened a credit card using my social security number",
    "My identity was stolen and now there are accounts on my credit report that I never opened",
    "Fraudulent accounts appearing on my credit report due to identity theft",
    "I have evidence that someone is using my name and address to obtain credit illegally",
    "Identity theft victim needing fraud alerts and account removal from credit bureaus",
    "There are unauthorized inquiries on my credit report from companies I never contacted",
    "Someone used my information to file a false tax return and now it appears on my credit",

    # Debt validation / collection
    "I demand validation of this debt under the FDCPA before you continue collection efforts",
    "A debt collector keeps calling me about a debt I do not recognize",
    "Please validate the debt you claim I owe and provide proof of the original contract",
    "I am being harassed by a collection agency for a medical bill I already paid",
    "Debt collector reported a debt to credit bureaus without validating it first",
    "I requested debt validation but the collector never responded and continues to call",
    "Collection agency added unauthorized fees to my original debt amount",
    "I do not owe this debt and demand proof of the agreement with my signature",
    "A third party debt collector is trying to collect on an old debt that is past the statute of limitations",
    "Debt collector called my workplace after I told them not to contact me there",

    # Credit report errors
    "There are errors on my credit report that need to be investigated and corrected",
    "My credit report shows accounts that do not belong to me",
    "The credit bureau refuses to investigate the errors I reported on my credit file",
    "Incorrect information on my credit report is lowering my credit score unfairly",
    "I disputed errors on my credit report but the bureau marked them as verified without investigation",
    "My credit report contains inaccurate late payment information that I want removed",
    "The credit reporting agency did not respond to my dispute within 30 days as required by law",
    "There is a collection account on my report that was paid off years ago but still shows as unpaid",
    "Multiple accounts on my credit report have incorrect balances and account statuses",
    "Credit report shows a bankruptcy that I never filed",

    # Credit card issues
    "I was charged an annual fee that I never agreed to on my credit card",
    "My credit card company increased my interest rate without notifying me",
    "I am being charged late fees even though I made my payment on time",
    "The credit card company closed my account without explanation",
    "I was charged a foreign transaction fee for a purchase made in the United States",
    "Unauthorized recurring charges on my credit card that the issuer refuses to remove",
    "Credit card rewards were revoked without notice or explanation",
    "My credit limit was reduced without any prior notification",
    "The bank is reporting my credit card as delinquent when my payments are current",

    # Mortgage / housing
    "My mortgage servicer applied my payment to the wrong account",
    "I am having trouble getting a loan modification despite meeting all requirements",
    "My escrow account balance is incorrect and the servicer will not correct it",
    "The mortgage company is charging improper fees on my loan statement",
    "I was not given proper notice before foreclosure proceedings began",
    "My mortgage payment was not credited on time causing a late fee",
    "The lender failed to provide proper payoff statement within the required timeframe",
    "Property tax paid from escrow was not applied correctly and now I have a tax lien",
    "I requested a forbearance plan but the mortgage company denied it without reason",

    # Student loans
    "My student loan servicer will not process my income-driven repayment application",
    "I am being charged excessive fees on my student loan account",
    "The student loan company reported me as delinquent even though I am in forbearance",
    "My loan servicer lost my paperwork for loan consolidation",
    "I am having trouble getting my Public Service Loan Forgiveness application processed",
    "Student loan payments were not applied correctly causing my balance to increase",
    "The servicer is trying to collect on a student loan that was discharged in bankruptcy",
    "I was misled about the terms of my student loan repayment plan",
    "My student loan interest rate was changed without notification",

    # Bank accounts / transactions
    "My bank account was charged without my authorization for a subscription I never signed up for",
    "The bank charged me multiple overdraft fees for a single transaction",
    "My debit card was charged for a purchase I did not make and the bank will not refund it",
    "The bank closed my account without warning and will not return my money",
    "A check I deposited was held for an unreasonable amount of time",
    "The bank is charging monthly maintenance fees that were supposed to be waived",
    "Someone used my debit card information to make unauthorized purchases online",
    "My account was charged for ATM fees at the bank's own ATM",
    "The bank refused to reverse a fraudulent transaction despite providing evidence",
    "Direct deposit was not credited to my account on the expected pay date",

    # Auto loans
    "My car loan was reported as repossessed even though I am current on payments",
    "The auto lender is charging unreasonable fees for loan payoff",
    "I was misled about the interest rate on my car loan at the time of purchase",
    "The car dealership added extra products to my loan without my consent",
    "Auto loan company reported a late payment that was actually paid on time",
    "My vehicle was repossessed without proper legal notice",
    "The lender refuses to release the title after I paid off my auto loan",

    # Debt collection harassment
    "A debt collector is calling me multiple times a day including before 8 am",
    "The collection agency threatened to sue me for a debt that is past the statute of limitations",
    "Collector contacted my family members about a debt which is a violation of my rights",
    "The debt collector refuses to provide validation of the debt they are trying to collect",
    "I told the collector to stop calling but they continue to harass me daily",
    "A collection agency is trying to collect a debt that was already paid in full",

    # Various
    "I was denied credit because of an error on my credit report that I have been trying to fix",
    "My prepaid card was charged fees that were not disclosed when I purchased it",
    "The wire transfer I sent never arrived at its destination and the company will not help",
    "I am having trouble accessing my online banking account for over a week",
    "Check deposit was delayed and caused my other payments to bounce",
    "The bank is reporting me to ChexSystems for an account that was closed in good standing",

    # Additional short variations
    "My credit score dropped because of inaccurate information on my credit report",
    "I need help disputing fraudulent charges on my bank statement",
    "The lender will not work with me on a payment plan despite my hardship",
    "My personal information was exposed in a data breach and now accounts are being opened",
    "I was charged for credit monitoring I never signed up for",
    "The company continues to report a closed account as open on my credit report",
    "I need to place a fraud alert on my credit file due to identity theft",
    "The bank is charging me interest on a zero balance account",
    "My mortgage company lost my insurance information and force-placed expensive insurance",
    "I am being billed for services I never received from this company",
]


def main():
    config = load_config()
    root = get_project_root()

    TOP_N = 1
    inf_cfg = config["inference"]
    emb_cfg = config["embedding"]
    paths = config["paths"]

    # ── Load artifacts ─────────────────────────────────
    print("Loading topic lookup...")
    lookup = load_lookup(root, inf_cfg["lookup_path"])

    print("Loading centroids...")
    centroids_path = root / paths["topic_centroids"]
    ensure_file_exists(centroids_path, "Topic centroids")
    centroids = np.load(centroids_path, allow_pickle=True).item()

    # ── Load models ────────────────────────────────────
    print("Loading F2LLM embedding model...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(
        emb_cfg["model_name"],
        trust_remote_code=True,
    )

    print("Loading BERTopic model...")
    from bertopic import BERTopic
    model_dir = root / inf_cfg["model_dir"]
    bertopic_model = BERTopic.load(model_dir, embedding_model=embedding_model)

    # ── Run inference on all test complaints ────────────
    print(f"\nTesting {len(TEST_COMPLAINTS)} complaints...\n")

    results = []
    for i, text in enumerate(TEST_COMPLAINTS):
        # Centroid Similarity
        centroid_preds = predict_with_centroids(
            text, embedding_model, centroids, lookup, TOP_N
        )
        centroid_id = centroid_preds[0].topic_id if centroid_preds else -2
        centroid_score = round(centroid_preds[0].score, 4) if centroid_preds else 0.0
        centroid_label = lookup.get(centroid_id, "Unknown")

        # BERTopic Transform
        bt_topics, bt_probs = bertopic_model.transform([text])
        bt_id = int(bt_topics[0])
        bt_label = lookup.get(bt_id, "Unknown")
        try:
            if bt_probs is not None and len(bt_probs) > 0 and len(bt_probs[0]) > bt_topics[0]:
                bt_prob = round(float(bt_probs[0][bt_topics[0]]), 4)
            else:
                bt_prob = 0.0
        except (TypeError, IndexError):
            bt_prob = 0.0

        match = "YES" if centroid_id == bt_id else "NO"

        results.append({
            "Index": i + 1,
            "Complaint": text[:70],
            "BERTopic_ID": bt_id,
            "BERTopic_Label": bt_label,
            "Centroid_ID": centroid_id,
            "Centroid_Label": centroid_label,
            "Centroid_Score": centroid_score,
            "Match": match,
        })

    # ── Summary ────────────────────────────────────────
    summary = pd.DataFrame(results)

    match_count = np.sum(summary["Match"] == "YES")
    total = len(summary)
    match_rate = match_count / total * 100

    print("=" * 110)
    print("AGREEMENT TEST RESULTS")
    print("=" * 110)
    print(f"{'Index':<6} {'Complaint':<50} {'BERTopic':<8} {'Centroid':<8} {'Match'}")
    print("-" * 110)
    for _, row in summary.iterrows():
        match_char = "✓" if row["Match"] == "YES" else "✗"
        print(f"{row['Index']:<6} {row['Complaint']:<50} {row['BERTopic_ID']:<8} {row['Centroid_ID']:<8} {match_char}")

    print("=" * 110)
    print(f"Total: {total} complaints")
    print(f"Agreements: {match_count} ({match_rate:.1f}%)")
    print(f"Disagreements: {total - match_count} ({100 - match_rate:.1f}%)")
    print()

    # List disagreements
    disagreements = summary[summary["Match"] == "NO"]
    if len(disagreements) > 0:
        print("Disagreements (BERTopic != Centroid):")
        print("-" * 110)
        for _, row in disagreements.iterrows():
            print(f"  #{row['Index']}: \"{row['Complaint']}\"")
            print(f"     BERTopic:  Topic {row['BERTopic_ID']} — {row['BERTopic_Label']}")
            print(f"     Centroid:  Topic {row['Centroid_ID']} — {row['Centroid_Label']} (score={row['Centroid_Score']})")
            print()

    # Save full results
    output_path = root / "outputs" / "agreement_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Full results saved: {output_path}")


if __name__ == "__main__":
    main()
