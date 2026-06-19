<img width="2300" height="742" alt="image" src="https://github.com/user-attachments/assets/4a9ffcbc-7892-41ad-9136-23218753154f" />

<img width="2220" height="2601" alt="image" src="https://github.com/user-attachments/assets/662bea0e-8097-41a0-98f8-7a44d20875b9" />

# Comment Analyzer: Comment Management System

An AI-powered customer support automation system for **Instagram and Facebook comments/messages**.  
It classifies customer queries, detects priority, routes them to the right team, and generates reply suggestions using a fine-tuned **DistilBERT** model and **Google Gemini**.

---

## Overview

Businesses receive a large number of comments and DMs every day:

- product inquiries
- price questions
- complaints
- refund requests
- order-related queries
- spam / irrelevant messages

Manually handling everything is slow and inconsistent.  
This project creates a practical workflow to **sort, classify, prioritize, and respond** to customer messages faster.

---

## Features

### Intent Classification
Classifies comments into business categories such as:

- `account_issue`
- `customer_support`
- `feedback`
- `order_management`
- `payment_issue`
- `product_inquiry`
- `refund_return`
- `shipping_delivery`

### Priority Detection
Automatically detects high-priority issues like:

- refund requests
- fraud complaints
- scam reports
- damaged products
- urgent queries

### Team Routing
Routes each query to the correct team:

| Intent | Assigned Team |
|--------|---------------|
| `refund_return` | `complaint_team` |
| `product_inquiry` | `sales_team` |
| `account_issue` | `customer_support` |
| `shipping_delivery` | `customer_support` |

### AI Reply Generation
Uses **Google Gemini** to generate professional and human-like response suggestions.

### Analytics Dashboard
Shows:

- total comments
- refund cases
- sales leads
- high-priority cases
- intent distribution
- priority distribution
- team routing distribution

### Bulk CSV Processing
Upload a CSV file and process multiple comments at once.

### Downloadable Output
Export processed results as a CSV file.

---

## System Architecture

```text
Customer Comment
        ↓
Streamlit Web App
        ↓
Text Preprocessing
        ↓
DistilBERT Intent Classifier
        ↓
Priority Detection
        ↓
Team Routing
        ↓
Gemini Reply Generator
        ↓
Analytics Dashboard / CSV Export
