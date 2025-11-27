# Restaurant Service Flow

The following Mermaid diagram illustrates the complete service flow for both **Dine-in** and **Takeaway** customers, from arrival to payment and departure.

## Flow Diagram

```mermaid
graph TD
    subgraph Dine-in Flow
        A[START] --> B(Customer Arrives)
        B --> C{Dine-in?}
        C -- Yes --> D[Seated & Menu Given]
        D --> E[Order from QR Menu]
        E --> F[Waiting for Food]
        F --> G[Receive Food]
        G --> H[Customers Eat]
    end

    subgraph Takeaway Flow
        C -- No --> J[Go to Takeaway Counter]
        J --> K[Order from QR Menu]
        K --> L[Waiting for Food]
        L --> M[Receive Food]
    end

    H --> I[Make Payment at POS]
    M --> I

    I --> N{Select Payment Method?}

    N -- 1. Cash --> O[Process Cash Payment]
    N -- 2. Card --> P_Card[Process Card Payment Tap or Chip]
    N -- 3. QR Code --> Q[Process QR Code Payment]

    O --> R[Customer Departs]
    P_Card --> R
    Q --> R

    R --> S[END]



