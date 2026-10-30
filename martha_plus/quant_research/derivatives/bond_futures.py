"""Government-bond futures invoice, basis and cheapest-to-deliver diagnostics."""
from __future__ import annotations
from typing import Sequence
from ..core import finite_series


def invoice_price(futures_price: float, conversion_factor: float,
                  accrued_interest: float = 0.0, face_value: float = 100.0) -> float:
    if futures_price<=0 or conversion_factor<=0 or face_value<=0 or accrued_interest<0:
        raise ValueError("invalid futures inputs")
    return futures_price*conversion_factor*face_value/100.0+accrued_interest


def net_basis(cash_clean_price: float, accrued_interest: float,
              futures_price: float, conversion_factor: float) -> float:
    if min(cash_clean_price,futures_price,conversion_factor)<=0 or accrued_interest<0: raise ValueError("invalid prices")
    dirty=cash_clean_price+accrued_interest
    return dirty-futures_price*conversion_factor


def implied_repo_rate(cash_clean_price: float, accrued_interest: float,
                      futures_price: float, conversion_factor: float,
                      days_to_delivery: float, coupon_cashflows: Sequence[tuple[float,float]] = ()) -> float:
    """Simple annualized implied repo, with coupon cashflows before delivery."""
    if days_to_delivery<=0: raise ValueError("delivery days must be positive")
    purchase=cash_clean_price+accrued_interest
    delivery=futures_price*conversion_factor+accrued_interest
    reinvested=sum(amount*(1.0+0.0*days/days_to_delivery) for days,amount in coupon_cashflows if 0<=days<=days_to_delivery)
    return ((delivery+reinvested)/purchase-1.0)*360.0/days_to_delivery


def cheapest_to_deliver(cash_prices: Sequence[float], accrued: Sequence[float],
                        conversion_factors: Sequence[float], futures_price: float,
                        repo_rates: Sequence[float] | None = None) -> dict:
    """Select lowest net-basis bond; if repo rates supplied, rank by implied repo."""
    p=finite_series(cash_prices); ai=finite_series(accrued); cf=finite_series(conversion_factors)
    if not(len(p)==len(ai)==len(cf)) or any(v<=0 for v in p+cf) or any(v<0 for v in ai): raise ValueError("bond inputs invalid")
    basis=[net_basis(p[i],ai[i],futures_price,cf[i]) for i in range(len(p))]
    if repo_rates is None:
        idx=min(range(len(p)),key=basis.__getitem__); ranking="net_basis"
    else:
        repo=finite_series(repo_rates)
        if len(repo)!=len(p): raise ValueError("repo rates length mismatch")
        idx=max(range(len(p)),key=repo.__getitem__); ranking="implied_repo"
    return {"ctd_index":idx,"net_basis":tuple(basis),"ranking":ranking,
            "best_value":basis[idx] if ranking=="net_basis" else repo[idx],
            "warning":"simplified basis/CTD ranking; include financing, coupon reinvestment and delivery options for production work"}


def conversion_factor(price_at_standard_yield: float, face_value: float = 100.0) -> float:
    if price_at_standard_yield<=0 or face_value<=0: raise ValueError("prices must be positive")
    return price_at_standard_yield/face_value
