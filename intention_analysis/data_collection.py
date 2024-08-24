import pandas as pd

def load_data():
    data = {
    "text": [
        "I want to buy a new laptop",
        "Can you help me with my order?",
        "I need a refund for my purchase",
        "Where is my package?",
        "Tell me about the latest deals",
        "I am looking for a new smartphone",
        "My order hasn't arrived yet",
        "How do I return an item?",
        "What is the status of my delivery?",
        "Show me some discounts on electronics",
        "I would like to cancel my subscription",
        "I have a question about my account",
        "I need assistance with my account settings",
        "Can I exchange my product?",
        "Why is my order delayed?",
        "What are today's special offers?",
        "I want to track my shipment",
        "Can I get a replacement for this item?",
        "How do I update my billing information?",
        "I need help with resetting my password",
        "What payment methods do you accept?",
        "My package was damaged when it arrived",
        "Can I get a refund for the defective product?",
        "I am interested in the latest promotions",
        "Please guide me through the return process",
        "I would like to know more about your services",
        "What are the terms for a refund?",
        "Can you provide tracking details?",
        "Is there any sale on home appliances?",
        "How do I apply for a refund?",
    ],
    "intention": [
        "Purchase", "Support", "Refund", "Tracking", "Inquiry",
        "Purchase", "Tracking", "Refund", "Tracking", "Inquiry",
        "Support", "Support", "Support", "Refund", "Tracking",
        "Inquiry", "Tracking", "Refund", "Support", "Support",
        "Inquiry", "Refund", "Refund", "Inquiry", "Support",
        "Inquiry", "Tracking", "Inquiry", "Refund"
    ],
}
       
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
