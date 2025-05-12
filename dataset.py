import pandas as pd
import random

# Original examples from the document
examples = [
    # Intent 1: Transaction_Date_navigator
    {
        "text": "Hi team, Can you pull together a schedule of important dates for the escrow process on the 125 King St deal? We're especially concerned with closing and due diligence periods. Thanks!",
        "intent": "Intent_Transaction_Date_navigator"
    },
    # Intent 2: Clause_Protect  
    {
        "text": "Hey, I'm reviewing the lease on the 3rd Avenue property. Can you check if there are any red flags—like missing indemnity clauses or unfavorable assignment terms?",
        "intent": "Intent_Clause_Protect"
    },
    # Intent 3: Lease_Abstraction
    {
        "text": "Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule.",
        "intent": "Intent_Lease_Abstraction"
    },
    # Intent 4: Comparison_LOI_Lease
    {
        "text": "Compare the signed lease to the LOI we submitted last month. I want to know what terms got changed or added, especially around TI allowances.",
        "intent": "Intent_Comparison_LOI_Lease"
    },
    # Intent 5: Company_research
    {
        "text": "Could you do a background check on Wexford Corp before we proceed? I'm particularly interested in any public disputes or bankruptcies in the past 5 years.",
        "intent": "Intent_Company_research"
    },
    # Intent 6: Amendment_Abstraction
    {
        "text": "Please summarize the changes from the latest amendment to the original lease for the Grandview Tower property.",
        "intent": "Intent_Amendment_Abstraction"
    },
    # Intent 7: Sales_Listings_Comparison
    {
        "text": "We have three broker sales packages for the 42nd St building. Can you give me a side-by-side summary of pricing, cap rate, and avg. PSF?",
        "intent": "Intent_Sales_Listings_Comparison"
    },
    # Intent 8: Lease_Listings_Comparison
    {
        "text": "Compare the lease listings we received for Midtown West. Looking to identify which has the most favorable terms per sq ft.",
        "intent": "Intent_Lease_Listings_Comparison"
    }
]

# Create variations for each intent to reach 80+ total examples
variations = []

# 1. Transaction Date Navigator variations
transaction_variations = [
    "Need help creating a timeline for the Metro Tower closing - especially milestone dates and deadlines.",
    "Can someone create a calendar of key dates for our downtown property transaction? Closing is next month.",
    "Please extract all important dates from the Park Avenue deal documents - I need a complete schedule.",
    "What are the critical dates in our 55 Wall St transaction? Need to see escrow, inspection, and closing dates.",
    "Create a transaction calendar for the Broadway project showing all due dates and deadlines.",
    "I need a complete timeline for the Hudson Yards deal - particularly interested in financing deadlines.",
    "Extract all key dates from the office building purchase agreement - closing, contingencies, everything.",
    "Build me a schedule of important milestones for the retail space acquisition on 5th Avenue.",
    "What dates should we be tracking for the Chicago warehouse deal? Need a full timeline.",
    "Please compile all deadline dates for the Boston office transaction - closing is in 6 weeks."
]

# 2. Clause Protection variations
clause_variations = [
    "Review the downtown lease for problematic clauses - concerned about liability and maintenance terms.",
    "Can you identify any unfavorable terms in this lease? Especially looking at breach and penalty clauses.",
    "Check the Madison Avenue lease for missing protections - need to see if we have adequate coverage.",
    "Review the new lease draft for red flags, particularly around common area charges and escalations.",
    "Please examine this lease for risky provisions - especially interested in dispute resolution terms.",
    "I need a clause review on the Brooklyn warehouse lease - looking for gaps in tenant protections.",
    "Analyze the lease terms for potential issues - focus on subletting rights and assignment restrictions.",
    "Can you flag any concerning clauses in the Times Square office lease? Particularly around penalties.",
    "Check this lease for missing key protections - worried about maintenance responsibilities.",
    "Review the retail lease for unfavorable terms - need to identify any risky provisions."
]

# 3. Lease Abstraction variations
abstraction_variations = [
    "Extract key lease terms from the Park Place property - need rent, dates, and renewal options.",
    "Please summarize the main provisions of the attached office lease - particularly financial terms.",
    "Abstract the important details from the Cincinnati retail lease - base rent, escalations, options.",
    "Pull out key information from the Miami office lease - rent schedule, term, and tenant responsibilities.",
    "Need a lease summary for the Dallas warehouse - extract rent, dates, and critical provisions.",
    "Create an abstract of the Seattle office lease focusing on financial and term information.",
    "Extract key data points from the Portland retail lease - particularly interested in rent structure.",
    "Please abstract the essential terms from the San Francisco lease - need full financial details.",
    "Summarize the main provisions of the Phoenix office lease - rent, term, and options.",
    "Abstract the key terms from the Atlanta warehouse lease - focusing on rates and escalations."
]

# 4. LOI/Lease Comparison variations
comparison_variations = [
    "Compare the executed lease with our original LOI - identify what changed in negotiations.",
    "Need a side-by-side of the LOI and final lease - show me all modifications and additions.",
    "Analyze differences between the proposed LOI and signed lease - focus on financial terms.",
    "Compare our letter of intent with the final agreement - what got negotiated differently?",
    "Show me the changes between the LOI and executed lease for the State Street property.",
    "Need a comparison of LOI vs final lease - particularly around tenant improvement allowances.",
    "Identify deviations between our original intent and the final lease terms.",
    "Compare the initial LOI with the final document - highlight all term changes.",
    "Show what changed from LOI to lease execution - especially interested in rent concessions.",
    "Create a comparison between our letter of intent and the final signed lease agreement."
]

# 5. Company Research variations
research_variations = [
    "Research ABC Properties before we finalize - check financial stability and litigation history.",
    "Need due diligence on Manhattan Development Group - any red flags in their background?",
    "Investigate Global Real Estate Partners - particularly interested in recent court cases.",
    "Run a background check on Metro Commercial LLC - check for bankruptcies or disputes.",
    "Research the financial strength of Urban Properties Inc before we proceed with the lease.",
    "Check the track record of Downtown Ventures - any legal issues or payment problems?",
    "Need company research on Hudson Bay Realty - focus on their reputation and stability.",
    "Investigate Coastal Properties Group - particularly interested in any regulatory issues.",
    "Research Empire Property Management's history - check for any tenant disputes.",
    "Due diligence needed on Midwest Commercial Real Estate - look for financial red flags."
]

# 6. Amendment Abstraction variations
amendment_variations = [
    "Summarize what changed in the latest lease amendment for the Capitol building.",
    "Extract new terms from Amendment #3 - show me what differs from the original lease.",
    "Need a summary of changes in the recent lease modification - what's new or different?",
    "Abstract the key changes from the First Amendment to the Central Park lease.",
    "Please highlight what changed in this lease amendment versus the base agreement.",
    "Extract modifications from the latest amendment - focus on financial and term changes.",
    "Summarize alterations in the Second Amendment to the Beverly Hills office lease.",
    "What's different in this lease amendment? Need to see all changes from original.",
    "Abstract the changes in Amendment 4 - particularly interested in new restrictions.",
    "Review the lease amendment and highlight what provisions were added or modified."
]

# 7. Sales Listings Comparison variations
sales_variations = [
    "Compare these three office building listings - need pricing, cap rates, and PSF analysis.",
    "Analyze the different broker packages for the retail center - focus on valuation metrics.",
    "Compare these sales listings side-by-side - interested in price and square footage details.",
    "Need a comparison of these investment opportunities - show pricing and yield data.",
    "Analyze our three listing options for the warehouse - compare asking prices and cap rates.",
    "Compare the broker materials for the office tower - focus on financial metrics.",
    "Side-by-side analysis needed for these property listings - pricing and cash flow data.",
    "Compare offerings for the retail space - need cap rate and PSF comparisons.",
    "Analyze these three sales packages - particularly interested in pricing and NOI.",
    "Compare the investment summaries - show pricing, cap rates, and value per square foot."
]

# 8. Lease Listings Comparison variations
lease_variations = [
    "Compare available office spaces downtown - focus on lease terms and rates per square foot.",
    "Analyze these retail lease opportunities - which has the best terms and pricing?",
    "Compare the three industrial lease options - need rate and term comparisons.",
    "Side-by-side of these commercial leases - show rates, concessions, and key terms.",
    "Compare available spaces in midtown - focus on rent and lease flexibility.",
    "Analyze these office lease options - which offers the best value per square foot?",
    "Need comparison of these warehouse lease opportunities - rates and terms analysis.",
    "Compare the retail lease listings - interested in base rent and escalation clauses.",
    "Analyze available office spaces - compare lease rates and tenant improvement allowances.",
    "Compare these lease opportunities - show me the best terms per square foot."
]

# Compile all variations
intent_variations = {
    "Intent_Transaction_Date_navigator": transaction_variations,
    "Intent_Clause_Protect": clause_variations,
    "Intent_Lease_Abstraction": abstraction_variations,
    "Intent_Comparison_LOI_Lease": comparison_variations,
    "Intent_Company_research": research_variations,
    "Intent_Amendment_Abstraction": amendment_variations,
    "Intent_Sales_Listings_Comparison": sales_variations,
    "Intent_Lease_Listings_Comparison": lease_variations
}

# Create the full dataset
dataset = []

# Add original examples
for example in examples:
    dataset.append(example)

# Add variations
for intent, variations_list in intent_variations.items():
    for variation in variations_list:
        dataset.append({
            "text": variation,
            "intent": intent
        })

# Create DataFrame
df = pd.DataFrame(dataset)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display statistics
print("Dataset Statistics:")
print(f"Total examples: {len(df)}")
print("\nExamples per intent:")
print(df['intent'].value_counts())

# Save to CSV
df.to_csv('email_intent_dataset.csv', index=False)

# Display first few examples
print("\nFirst 5 examples:")
print(df.head())

# Preview the dataset structure
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())