"""
Frontend Pages Module
Contains all page components for the application
"""

# Import pages for easier access
try:
    from .dashboard import show_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

try:
    from .analytics import show_analytics
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

__all__ = [
    'show_dashboard',
    'show_analytics',
    'DASHBOARD_AVAILABLE',
    'ANALYTICS_AVAILABLE'
]