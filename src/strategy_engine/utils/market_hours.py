"""
Market Hours Validator for Strategy Engine
==========================================

This module provides utilities for validating market hours and trading time windows.
It handles Indian market hours and special trading days.
"""

from datetime import datetime, time

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MarketHoursValidator:
    """Validator for Indian market trading hours."""
    
    def __init__(self):
        """Initialize market hours validator."""
        # Indian market hours (NSE/BSE)
        self.market_start = time(9, 15)  # 9:15 AM
        self.market_end = time(15, 30)   # 3:30 PM
        self.pre_market_start = time(8, 45)  # 8:45 AM
        self.post_market_end = time(16, 30)  # 4:30 PM
        
        # Trading days (Monday=0, Sunday=6)
        self.trading_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
    def is_market_day(self) -> bool:
        """Check if today is a trading day."""
        today = datetime.now().weekday()
        return today in self.trading_days
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        if not self.is_market_day():
            return False
        
        current_time = datetime.now().time()
        return self.market_start <= current_time <= self.market_end
    
    def is_pre_market_time(self) -> bool:
        """Check if it's pre-market time."""
        if not self.is_market_day():
            return False
        
        current_time = datetime.now().time()
        return self.pre_market_start <= current_time < self.market_start
    
    def is_post_market_time(self) -> bool:
        """Check if it's post-market time."""
        if not self.is_market_day():
            return False
        
        current_time = datetime.now().time()
        return self.market_end < current_time <= self.post_market_end
    
    def get_market_status(self) -> str:
        """Get current market status."""
        if not self.is_market_day():
            return "CLOSED_HOLIDAY"
        elif self.is_pre_market_time():
            return "PRE_MARKET"
        elif self.is_market_hours():
            return "OPEN"
        elif self.is_post_market_time():
            return "POST_MARKET"
        else:
            return "CLOSED"
    
    def time_until_market_open(self) -> int:
        """Get minutes until market opens."""
        if self.is_market_hours():
            return 0
        
        now = datetime.now()
        
        # If it's after market hours today, calculate for next trading day
        if now.time() > self.market_end:
            # Find next trading day
            next_day = now.replace(hour=9, minute=15, second=0, microsecond=0)
            next_day = next_day.replace(day=next_day.day + 1)
            
            while next_day.weekday() not in self.trading_days:
                next_day = next_day.replace(day=next_day.day + 1)
        else:
            # Market opens today
            next_day = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        delta = next_day - now
        return int(delta.total_seconds() / 60)
    
    def time_until_market_close(self) -> int:
        """Get minutes until market closes."""
        if not self.is_market_hours():
            return 0
        
        now = datetime.now()
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        delta = market_close - now
        return int(delta.total_seconds() / 60)
