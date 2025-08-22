try:
    from .xgb_model import XGBDirectionModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .fallback_model import FallbackDirectionModel