# import pandas as pd
# import random

# # Original examples from the document
# examples = [
#     # Intent 1: Transaction_Date_navigator
#     {
#         "text": "Hi team, Can you pull together a schedule of important dates for the escrow process on the 125 King St deal? We're especially concerned with closing and due diligence periods. Thanks!",
#         "intent": "Intent_Transaction_Date_navigator"
#     },
#     # Intent 2: Clause_Protect  
#     {
#         "text": "Hey, I'm reviewing the lease on the 3rd Avenue property. Can you check if there are any red flags—like missing indemnity clauses or unfavorable assignment terms?",
#         "intent": "Intent_Clause_Protect"
#     },
#     # Intent 3: Lease_Abstraction
#     {
#         "text": "Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule.",
#         "intent": "Intent_Lease_Abstraction"
#     },
#     # Intent 4: Comparison_LOI_Lease
#     {
#         "text": "Compare the signed lease to the LOI we submitted last month. I want to know what terms got changed or added, especially around TI allowances.",
#         "intent": "Intent_Comparison_LOI_Lease"
#     },
#     # Intent 5: Company_research
#     {
#         "text": "Could you do a background check on Wexford Corp before we proceed? I'm particularly interested in any public disputes or bankruptcies in the past 5 years.",
#         "intent": "Intent_Company_research"
#     },
#     # Intent 6: Amendment_Abstraction
#     {
#         "text": "Please summarize the changes from the latest amendment to the original lease for the Grandview Tower property.",
#         "intent": "Intent_Amendment_Abstraction"
#     },
#     # Intent 7: Sales_Listings_Comparison
#     {
#         "text": "We have three broker sales packages for the 42nd St building. Can you give me a side-by-side summary of pricing, cap rate, and avg. PSF?",
#         "intent": "Intent_Sales_Listings_Comparison"
#     },
#     # Intent 8: Lease_Listings_Comparison
#     {
#         "text": "Compare the lease listings we received for Midtown West. Looking to identify which has the most favorable terms per sq ft.",
#         "intent": "Intent_Lease_Listings_Comparison"
#     }
# ]

# # Create variations for each intent to reach 80+ total examples
# variations = []

# # 1. Transaction Date Navigator variations
# transaction_variations = [
#     "Need help creating a timeline for the Metro Tower closing - especially milestone dates and deadlines.",
#     "Can someone create a calendar of key dates for our downtown property transaction? Closing is next month.",
#     "Please extract all important dates from the Park Avenue deal documents - I need a complete schedule.",
#     "What are the critical dates in our 55 Wall St transaction? Need to see escrow, inspection, and closing dates.",
#     "Create a transaction calendar for the Broadway project showing all due dates and deadlines.",
#     "I need a complete timeline for the Hudson Yards deal - particularly interested in financing deadlines.",
#     "Extract all key dates from the office building purchase agreement - closing, contingencies, everything.",
#     "Build me a schedule of important milestones for the retail space acquisition on 5th Avenue.",
#     "What dates should we be tracking for the Chicago warehouse deal? Need a full timeline.",
#     "Please compile all deadline dates for the Boston office transaction - closing is in 6 weeks."
# ]

# # 2. Clause Protection variations
# clause_variations = [
#     "Review the downtown lease for problematic clauses - concerned about liability and maintenance terms.",
#     "Can you identify any unfavorable terms in this lease? Especially looking at breach and penalty clauses.",
#     "Check the Madison Avenue lease for missing protections - need to see if we have adequate coverage.",
#     "Review the new lease draft for red flags, particularly around common area charges and escalations.",
#     "Please examine this lease for risky provisions - especially interested in dispute resolution terms.",
#     "I need a clause review on the Brooklyn warehouse lease - looking for gaps in tenant protections.",
#     "Analyze the lease terms for potential issues - focus on subletting rights and assignment restrictions.",
#     "Can you flag any concerning clauses in the Times Square office lease? Particularly around penalties.",
#     "Check this lease for missing key protections - worried about maintenance responsibilities.",
#     "Review the retail lease for unfavorable terms - need to identify any risky provisions."
# ]

# # 3. Lease Abstraction variations
# abstraction_variations = [
#     "Extract key lease terms from the Park Place property - need rent, dates, and renewal options.",
#     "Please summarize the main provisions of the attached office lease - particularly financial terms.",
#     "Abstract the important details from the Cincinnati retail lease - base rent, escalations, options.",
#     "Pull out key information from the Miami office lease - rent schedule, term, and tenant responsibilities.",
#     "Need a lease summary for the Dallas warehouse - extract rent, dates, and critical provisions.",
#     "Create an abstract of the Seattle office lease focusing on financial and term information.",
#     "Extract key data points from the Portland retail lease - particularly interested in rent structure.",
#     "Please abstract the essential terms from the San Francisco lease - need full financial details.",
#     "Summarize the main provisions of the Phoenix office lease - rent, term, and options.",
#     "Abstract the key terms from the Atlanta warehouse lease - focusing on rates and escalations."
# ]

# # 4. LOI/Lease Comparison variations
# comparison_variations = [
#     "Compare the executed lease with our original LOI - identify what changed in negotiations.",
#     "Need a side-by-side of the LOI and final lease - show me all modifications and additions.",
#     "Analyze differences between the proposed LOI and signed lease - focus on financial terms.",
#     "Compare our letter of intent with the final agreement - what got negotiated differently?",
#     "Show me the changes between the LOI and executed lease for the State Street property.",
#     "Need a comparison of LOI vs final lease - particularly around tenant improvement allowances.",
#     "Identify deviations between our original intent and the final lease terms.",
#     "Compare the initial LOI with the final document - highlight all term changes.",
#     "Show what changed from LOI to lease execution - especially interested in rent concessions.",
#     "Create a comparison between our letter of intent and the final signed lease agreement."
# ]

# # 5. Company Research variations
# research_variations = [
#     "Research ABC Properties before we finalize - check financial stability and litigation history.",
#     "Need due diligence on Manhattan Development Group - any red flags in their background?",
#     "Investigate Global Real Estate Partners - particularly interested in recent court cases.",
#     "Run a background check on Metro Commercial LLC - check for bankruptcies or disputes.",
#     "Research the financial strength of Urban Properties Inc before we proceed with the lease.",
#     "Check the track record of Downtown Ventures - any legal issues or payment problems?",
#     "Need company research on Hudson Bay Realty - focus on their reputation and stability.",
#     "Investigate Coastal Properties Group - particularly interested in any regulatory issues.",
#     "Research Empire Property Management's history - check for any tenant disputes.",
#     "Due diligence needed on Midwest Commercial Real Estate - look for financial red flags."
# ]

# # 6. Amendment Abstraction variations
# amendment_variations = [
#     "Summarize what changed in the latest lease amendment for the Capitol building.",
#     "Extract new terms from Amendment #3 - show me what differs from the original lease.",
#     "Need a summary of changes in the recent lease modification - what's new or different?",
#     "Abstract the key changes from the First Amendment to the Central Park lease.",
#     "Please highlight what changed in this lease amendment versus the base agreement.",
#     "Extract modifications from the latest amendment - focus on financial and term changes.",
#     "Summarize alterations in the Second Amendment to the Beverly Hills office lease.",
#     "What's different in this lease amendment? Need to see all changes from original.",
#     "Abstract the changes in Amendment 4 - particularly interested in new restrictions.",
#     "Review the lease amendment and highlight what provisions were added or modified."
# ]

# # 7. Sales Listings Comparison variations
# sales_variations = [
#     "Compare these three office building listings - need pricing, cap rates, and PSF analysis.",
#     "Analyze the different broker packages for the retail center - focus on valuation metrics.",
#     "Compare these sales listings side-by-side - interested in price and square footage details.",
#     "Need a comparison of these investment opportunities - show pricing and yield data.",
#     "Analyze our three listing options for the warehouse - compare asking prices and cap rates.",
#     "Compare the broker materials for the office tower - focus on financial metrics.",
#     "Side-by-side analysis needed for these property listings - pricing and cash flow data.",
#     "Compare offerings for the retail space - need cap rate and PSF comparisons.",
#     "Analyze these three sales packages - particularly interested in pricing and NOI.",
#     "Compare the investment summaries - show pricing, cap rates, and value per square foot."
# ]

# # 8. Lease Listings Comparison variations
# lease_variations = [
#     "Compare available office spaces downtown - focus on lease terms and rates per square foot.",
#     "Analyze these retail lease opportunities - which has the best terms and pricing?",
#     "Compare the three industrial lease options - need rate and term comparisons.",
#     "Side-by-side of these commercial leases - show rates, concessions, and key terms.",
#     "Compare available spaces in midtown - focus on rent and lease flexibility.",
#     "Analyze these office lease options - which offers the best value per square foot?",
#     "Need comparison of these warehouse lease opportunities - rates and terms analysis.",
#     "Compare the retail lease listings - interested in base rent and escalation clauses.",
#     "Analyze available office spaces - compare lease rates and tenant improvement allowances.",
#     "Compare these lease opportunities - show me the best terms per square foot."
# ]

# # Compile all variations
# intent_variations = {
#     "Intent_Transaction_Date_navigator": transaction_variations,
#     "Intent_Clause_Protect": clause_variations,
#     "Intent_Lease_Abstraction": abstraction_variations,
#     "Intent_Comparison_LOI_Lease": comparison_variations,
#     "Intent_Company_research": research_variations,
#     "Intent_Amendment_Abstraction": amendment_variations,
#     "Intent_Sales_Listings_Comparison": sales_variations,
#     "Intent_Lease_Listings_Comparison": lease_variations
# }

# # Create the full dataset
# dataset = []

# # Add original examples
# for example in examples:
#     dataset.append(example)

# # Add variations
# for intent, variations_list in intent_variations.items():
#     for variation in variations_list:
#         dataset.append({
#             "text": variation,
#             "intent": intent
#         })

# # Create DataFrame
# df = pd.DataFrame(dataset)

# # Shuffle the dataset
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Display statistics
# print("Dataset Statistics:")
# print(f"Total examples: {len(df)}")
# print("\nExamples per intent:")
# print(df['intent'].value_counts())

# # Save to CSV
# df.to_csv('email_intent_dataset.csv', index=False)

# # Display first few examples
# print("\nFirst 5 examples:")
# print(df.head())

# # Preview the dataset structure
# print("\nDataset shape:", df.shape)
# print("\nColumn names:", df.columns.tolist())

import pandas as pd
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from typing import List, Dict, Tuple
import itertools

# Original examples from the document
base_examples = [
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

# Additional seed examples for each intent to create more variety
additional_seeds = {
    "Intent_Transaction_Date_navigator": [
        "Need a timeline for all key dates in the property closing - escrow, inspections, and final walkthrough.",
        "Can you create a schedule of important milestones for the real estate transaction?",
        "Please extract all deadline dates from the purchase agreement - we need to track every step.",
        "What are the critical dates we need to monitor for the office building acquisition?",
        "Help me organize all the transaction dates for the retail space deal in downtown.",
    ],
    "Intent_Clause_Protect": [
        "Review the lease for any problematic clauses - particularly around liability and insurance.",
        "Check if there are adequate tenant protections in this lease agreement.",
        "I'm concerned about the penalty clauses in section 7 - please review.",
        "Identify any unfavorable terms that could affect our liability exposure.",
        "Please flag any missing or weak indemnification clauses in the lease.",
    ],
    "Intent_Lease_Abstraction": [
        "Extract the key financial terms from this lease - rent, escalations, and fees.",
        "Please summarize the main provisions of the retail lease agreement.",
        "Need a complete abstract of the office lease focusing on term and rent details.",
        "Pull out all important dates and financial obligations from this lease.",
        "Create a summary of the warehouse lease highlighting critical terms.",
    ],
    "Intent_Comparison_LOI_Lease": [
        "Show me what changed between our LOI and the final lease agreement.",
        "Compare the negotiated lease with our original letter of intent.",
        "I need to see all modifications made from LOI to final execution.",
        "What terms were added or changed during lease negotiations?",
        "Identify differences between the proposed and executed lease terms.",
    ],
    "Intent_Company_research": [
        "Research the landlord's company - any litigation or financial issues?",
        "Need due diligence on the tenant's corporate background and stability.",
        "Check if there are any red flags about this real estate company.",
        "Investigate the property management company's track record.",
        "Research the development firm before we proceed with negotiations.",
    ],
    "Intent_Amendment_Abstraction": [
        "Summarize what changed in the recent lease amendment.",
        "Extract the new terms from Amendment #2 to the original agreement.",
        "What modifications were made in the latest lease amendment?",
        "Please highlight changes in the lease amendment focused on rent adjustments.",
        "Compare the amendment against the base lease to show differences.",
    ],
    "Intent_Sales_Listings_Comparison": [
        "Compare the three property listings - pricing, cap rates, and PSF analysis.",
        "Analyze the different sales packages for the commercial building.",
        "Need a side-by-side comparison of these investment opportunities.",
        "Compare the broker packages focusing on financial metrics.",
        "Analyze these three listings for the best investment value.",
    ],
    "Intent_Lease_Listings_Comparison": [
        "Compare available office spaces - rates and lease terms analysis.",
        "Which of these retail spaces offers the best lease terms?",
        "Analyze the warehouse lease options for the most favorable conditions.",
        "Compare these commercial lease opportunities - rates and concessions.",
        "Show me the best lease terms among these available spaces.",
    ]
}

class EnhancedDatasetGenerator:
    def __init__(self):
        """Initialize augmentation tools for large-scale dataset generation"""
        print("Initializing enhanced dataset generator...")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Initialize augmenters
        try:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
        except Exception as e:
            print(f"Warning: Could not load synonym augmenter: {e}")
            self.synonym_aug = None
        
        try:
            self.keyboard_aug = nac.KeyboardAug(aug_p=0.1)
        except Exception as e:
            print(f"Warning: Could not load keyboard augmenter: {e}")
            self.keyboard_aug = None
        
        # Multi-intent connectors for natural combinations
        self.connectors = [
            " Also, ",
            " Additionally, ",
            " Plus, ",
            " Furthermore, ",
            " While we're at it, ",
            " And after that, ",
            " On top of that, ",
            " Moreover, ",
            " Besides that, ",
            " In addition, ",
        ]
    
    def augment_text(self, text: str, num_variations: int = 10) -> List[str]:
        """Generate variations of the input text using multiple augmentation techniques"""
        variations = [text]  # Always include original
        
        for _ in range(num_variations * 2):  # Generate more attempts for better variety
            augmented = text
            
            # Apply synonym replacement
            if self.synonym_aug and random.random() < 0.7:
                try:
                    aug_result = self.synonym_aug.augment(text)
                    if isinstance(aug_result, list):
                        aug_result = aug_result[0]
                    if aug_result != text and aug_result:
                        augmented = aug_result
                except:
                    pass
            
            # Apply minor keyboard errors
            if self.keyboard_aug and random.random() < 0.3:
                try:
                    aug_result = self.keyboard_aug.augment(augmented)
                    if isinstance(aug_result, list):
                        aug_result = aug_result[0]
                    if aug_result != augmented and aug_result:
                        augmented = aug_result
                except:
                    pass
            
            # Word order variations
            if random.random() < 0.2:
                words = augmented.split()
                if len(words) > 4:
                    # Swap adjacent words or move words around
                    if random.random() < 0.5:
                        # Swap adjacent words
                        idx = random.randint(0, len(words) - 2)
                        words[idx], words[idx + 1] = words[idx + 1], words[idx]
                    else:
                        # Move a word to a different position
                        idx1 = random.randint(0, len(words) - 1)
                        idx2 = random.randint(0, len(words) - 1)
                        word = words.pop(idx1)
                        words.insert(idx2, word)
                    augmented = ' '.join(words)
            
            # Simple paraphrasing using word substitution
            if random.random() < 0.3:
                # Replace common words with alternatives
                replacements = {
                    'please': ['kindly', 'could you'],
                    'need': ['require', 'must have'],
                    'help': ['assist', 'support'],
                    'check': ['review', 'examine', 'verify'],
                    'show': ['display', 'provide', 'give'],
                    'create': ['generate', 'make', 'develop'],
                    'find': ['locate', 'identify', 'discover'],
                }
                
                for word, alternatives in replacements.items():
                    if word in augmented.lower():
                        replacement = random.choice(alternatives)
                        augmented = augmented.lower().replace(word, replacement)
                        augmented = augmented.capitalize()
            
            if augmented != text and augmented not in variations and len(variations) < num_variations:
                variations.append(augmented)
        
        return variations[:num_variations]
    
    def create_single_intent_examples(self, target_per_intent: int = 700) -> List[Dict]:
        """Create single-intent examples with heavy augmentation"""
        all_examples = []
        
        # Combine original and additional examples
        all_seeds = {}
        for example in base_examples:
            intent = example['intent']
            all_seeds[intent] = [example['text']]
        
        # Add additional seeds
        for intent, seeds in additional_seeds.items():
            all_seeds[intent].extend(seeds)
        
        # Generate variations for each intent
        for intent, texts in all_seeds.items():
            print(f"Generating single-intent examples for {intent}...")
            intent_examples = []
            
            # Add original examples
            for text in texts:
                intent_examples.append({
                    "text": text,
                    "intent": intent,
                    "type": "single"
                })
            
            # Generate many augmented versions
            variations_per_seed = target_per_intent // len(texts)
            
            for text in texts:
                variations = self.augment_text(text, num_variations=variations_per_seed)
                
                for var_text in variations:
                    if var_text != text and len(intent_examples) < target_per_intent:
                        intent_examples.append({
                            "text": var_text,
                            "intent": intent,
                            "type": "single"
                        })
            
            # Pad if needed with more variations of random seeds
            while len(intent_examples) < target_per_intent:
                seed_text = random.choice(texts)
                extra_variations = self.augment_text(seed_text, num_variations=10)
                for var in extra_variations:
                    if var not in [ex['text'] for ex in intent_examples] and len(intent_examples) < target_per_intent:
                        intent_examples.append({
                            "text": var,
                            "intent": intent,
                            "type": "single"
                        })
            
            all_examples.extend(intent_examples[:target_per_intent])
            print(f"Created {len(intent_examples[:target_per_intent])} single-intent examples for {intent}")
        
        return all_examples
    
    def create_multi_intent_examples(self, target_examples: int = 3000) -> List[Dict]:
        """Create realistic multi-intent examples"""
        all_examples = []
        all_intents = list(additional_seeds.keys())
        
        # Define logical intent combinations that make business sense
        logical_combinations = [
            # Common business combinations
            ("Intent_Lease_Abstraction", "Intent_Clause_Protect"),
            ("Intent_Comparison_LOI_Lease", "Intent_Amendment_Abstraction"),
            ("Intent_Company_research", "Intent_Clause_Protect"),
            ("Intent_Transaction_Date_navigator", "Intent_Clause_Protect"),
            ("Intent_Sales_Listings_Comparison", "Intent_Company_research"),
            ("Intent_Lease_Listings_Comparison", "Intent_Company_research"),
            ("Intent_Lease_Abstraction", "Intent_Amendment_Abstraction"),
            ("Intent_Transaction_Date_navigator", "Intent_Comparison_LOI_Lease"),
            # Three-intent combinations
            ("Intent_Lease_Abstraction", "Intent_Clause_Protect", "Intent_Company_research"),
            ("Intent_Transaction_Date_navigator", "Intent_Comparison_LOI_Lease", "Intent_Amendment_Abstraction"),
            ("Intent_Sales_Listings_Comparison", "Intent_Lease_Listings_Comparison", "Intent_Company_research"),
        ]
        
        print("Generating multi-intent examples...")
        
        # Create combinations for each logical pair/triplet
        for combination in logical_combinations:
            if len(combination) == 2:
                intent1, intent2 = combination
                examples_per_combo = target_examples // (len(logical_combinations) + 5)
                
                # Get seed texts for both intents
                seeds1 = base_examples + additional_seeds[intent1]
                seeds2 = base_examples + additional_seeds[intent2]
                
                for _ in range(examples_per_combo):
                    # Select random texts from each intent
                    text1 = random.choice([ex['text'] if isinstance(ex, dict) else ex for ex in seeds1 if (isinstance(ex, dict) and ex['intent'] == intent1) or (isinstance(ex, str))])
                    if isinstance(text1, dict):
                        text1 = text1['text']
                    
                    text2 = random.choice(additional_seeds[intent2])
                    
                    # Combine with a connector
                    connector = random.choice(self.connectors)
                    combined_text = text1 + connector + text2.lower()
                    
                    # Apply some augmentation to the combined text
                    augmented_versions = self.augment_text(combined_text, num_variations=3)
                    
                    for aug_text in augmented_versions:
                        all_examples.append({
                            "text": aug_text,
                            "intent": f"{intent1};{intent2}",
                            "type": "multi_2"
                        })
            
            elif len(combination) == 3:
                intent1, intent2, intent3 = combination
                examples_per_combo = target_examples // (len(logical_combinations) * 3)
                
                for _ in range(examples_per_combo):
                    # Get texts for all three intents
                    text1 = random.choice(additional_seeds[intent1])
                    text2 = random.choice(additional_seeds[intent2])
                    text3 = random.choice(additional_seeds[intent3])
                    
                    # Combine all three
                    connector1 = random.choice(self.connectors)
                    connector2 = random.choice(self.connectors)
                    combined_text = f"{text1}{connector1}{text2.lower()}{connector2}{text3.lower()}"
                    
                    # Apply light augmentation
                    augmented_versions = self.augment_text(combined_text, num_variations=2)
                    
                    for aug_text in augmented_versions:
                        all_examples.append({
                            "text": aug_text,
                            "intent": f"{intent1};{intent2};{intent3}",
                            "type": "multi_3"
                        })
        
        print(f"Created {len(all_examples)} multi-intent examples")
        return all_examples[:target_examples]
    
    def generate_large_dataset(self, single_intent_per_class: int = 700, multi_intent_total: int = 3000) -> pd.DataFrame:
        """Generate a large dataset with both single and multi-intent examples"""
        print("=== Generating Large-Scale Dataset ===")
        
        # Generate single-intent examples
        single_examples = self.create_single_intent_examples(single_intent_per_class)
        print(f"Total single-intent examples: {len(single_examples)}")
        
        # Generate multi-intent examples
        multi_examples = self.create_multi_intent_examples(multi_intent_total)
        print(f"Total multi-intent examples: {len(multi_examples)}")
        
        # Combine all examples
        all_examples = single_examples + multi_examples
        
        # Create DataFrame
        df = pd.DataFrame(all_examples)
        
        # Clean up the intent column for consistency
        df['intent'] = df['intent'].str.strip()
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df

# Main execution
if __name__ == "__main__":
    # Initialize enhanced generator
    generator = EnhancedDatasetGenerator()
    
    # Generate large dataset
    print("Creating large-scale dataset (target: ~10,000 examples)...")
    df = generator.generate_large_dataset(
        single_intent_per_class=700,  # 700 * 8 intents = 5,600 single-intent examples
        multi_intent_total=4400       # 4,400 multi-intent examples
    )
    
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {len(df)}")
    print(f"\nExamples by type:")
    print(df['type'].value_counts())
    
    # Show intent distribution
    print(f"\nSingle-intent distribution:")
    single_df = df[df['type'] == 'single']
    print(single_df['intent'].value_counts())
    
    print(f"\nMulti-intent examples (preview):")
    multi_df = df[df['type'].str.startswith('multi')]
    print(f"2-intent examples: {len(df[df['type'] == 'multi_2'])}")
    print(f"3-intent examples: {len(df[df['type'] == 'multi_3'])}")
    
    # Save to CSV
    df_output = df[['text', 'intent']]  # Keep only the columns needed for training
    df_output.to_csv('email_intent_dataset.csv', index=False)
    
    # Display sample examples
    print("\n=== Sample Examples ===")
    print("\nSingle-intent example:")
    print(df_output[df['type'] == 'single'].iloc[0]['text'][:100] + "...")
    print(f"Intent: {df_output[df['type'] == 'single'].iloc[0]['intent']}")
    
    print("\nMulti-intent example:")
    multi_sample = df_output[df['type'].str.startswith('multi')].iloc[0]
    print(multi_sample['text'][:100] + "...")
    print(f"Intent: {multi_sample['intent']}")
    
    print(f"\nDataset saved to 'email_intent_dataset.csv'")
    print(f"Final dataset shape: {df_output.shape}")
    print("\nDataset ready for multi-intent and single-intent training!")