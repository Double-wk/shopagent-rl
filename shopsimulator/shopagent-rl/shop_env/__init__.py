"""shop_A environment package: HTTP client + gym-like wrapper + segmented reward."""
from .client import ShopEnvClient
from .wrapper import ShopSimEnv, Action, parse_model_action
from . import reward

__all__ = ["ShopEnvClient", "ShopSimEnv", "Action", "parse_model_action", "reward"]
