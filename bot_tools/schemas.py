from pydantic import BaseModel, Field
from typing import Optional, List

# ---------------------------------------------------------
# 1. Check Account Status
# ---------------------------------------------------------
class CheckStatusIn(BaseModel):
    user_id: int

class Invoice(BaseModel):
    date: str
    amount: float

class CheckStatusOut(BaseModel):
    user_id: int
    username: str
    plan: str
    status: str
    last_invoice: Optional[Invoice] = None
    duration_s: float

# ---------------------------------------------------------
# 2. Upgrade Plan
# ---------------------------------------------------------
class UpgradePlanIn(BaseModel):
    user_id: int
    new_plan: str

class UpgradePlanOut(BaseModel):
    user_id: int
    old_plan: str
    new_plan: str
    status: str
    duration_s: float

# ---------------------------------------------------------
# 3. Get Order Status (NEW)
# ---------------------------------------------------------
class GetOrderStatusIn(BaseModel):
    order_id: int

class GetOrderStatusOut(BaseModel):
    order_id: int
    status: str
    order_date: str
    total_amount: float
    # We return a list of strings for simplicity in the demo
    items: List[str] = [] 
    duration_s: float

# ---------------------------------------------------------
# 4. Process Refund (UPDATED)
# ---------------------------------------------------------
class ProcessRefundIn(BaseModel):
    order_id: int
    # UPDATED: Reason is now compulsory (no longer Optional)
    reason: str = Field(..., description="The reason why the user wants a refund.")
    amount: Optional[float] = None
    idempotency_key: Optional[str] = None

class ProcessRefundOut(BaseModel):
    refund_id: int
    order_id: int
    amount: float
    status: str
    duration_s: float

# ---------------------------------------------------------
# 5. Handoff to Human
# ---------------------------------------------------------
class HandoffIn(BaseModel):
    conversation_id: str
    summary: str
    tags: List[str]

class HandoffOut(BaseModel):
    handoff_id: int
    status: str
    duration_s: float