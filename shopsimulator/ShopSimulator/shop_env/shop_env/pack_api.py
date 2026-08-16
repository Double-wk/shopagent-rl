import os
import sys
import logging
import time
from typing import Dict, Any, Optional, Set, List

import gym
from flask import Flask, request, jsonify, Response

sys.path.append("../")
from shop_agent import shop_agent
from web_agent_site.utils import DEBUG_PROD_SIZE
from web_agent_site.envs import WebAgentSiteEnv

# Constants
LOG_FILE = "shop_agent.log"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5
# `env16` is the standard deployment profile for this container.  It leaves
# enough headroom for the Python/JVM catalogue and the concurrent trainer.
# Set SHOP_ENV_MAX_NUM explicitly only when deliberately using another profile.
DEFAULT_ENV_MAX_NUM = int(os.environ.get("SHOP_ENV_MAX_NUM", "16"))
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

# Global variables
envs: List[Any] = []
free_env_index: Set[int] = set()
env_max_num: int = DEFAULT_ENV_MAX_NUM
# env_idx -> 上次活动 monotonic 时间。lease 超时回收用：被 SIGKILL 的采集 worker 持有的 env
# 不再发请求(无法 finally release_one)，超时后自动归还池，根治反复重启 collect 掏空 env 池。
env_assigned_time: Dict[int, float] = {}
ENV_LEASE_TIMEOUT = 900  # 15min: 单步 chat timeout 120s×5retry=600s，900s 安全覆盖活跃轨迹；超时=持有者已死

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/api/shop_agent', methods=['POST'])
def api_some_function() -> Response:
    """
    API endpoint for shop agent operations.
    
    Handles three types of actions:
    - release_all: Release all environments
    - release_one: Release a specific environment
    - reset/interact: Process shop agent actions
    
    Returns:
        JSON response with result or error message
    """
    data = request.json
    if data is None:
        logger.error("[Error] No JSON data provided in request")
        return jsonify({'result': {"error": "No JSON data provided"}})
    
    action = data.get('action')
    env_idx = data.get('env_idx', None)
    response = data.get('response', None)
    idx = data.get('idx', None)
    assigned = False  # 本次请求是否新分配了 env_idx（异常时据此归还，防止资源池泄漏）
    try:
        # Release all environments
        if action == 'release_all':
            for i in range(env_max_num):
                if i not in free_env_index:
                    free_env_index.add(i)
            env_assigned_time.clear()
            logger.info("[Init] All environments have been initialized")
            return jsonify({'result': {"message": "All environments have been initialized"}})

        # Release one environment
        if action == 'release_one':
            if env_idx is not None and isinstance(env_idx, int):
                if env_idx not in free_env_index:
                    free_env_index.add(env_idx)
                    env_assigned_time.pop(env_idx, None)
                    logger.info(f"[Release] Environment {env_idx} has been released")
                    return jsonify({'result': {"message": f"Environment {env_idx} has been released"}})
                else:
                    logger.warning(f"[Release] Environment {env_idx} is already free, no need to release again")
                    return jsonify({'result': {"message": f"Environment {env_idx} is already free"}})
            else:
                logger.error("[Error] No valid environment index provided")
                return jsonify({'result': {"error": "No valid environment index provided"}})

        # If env_idx is not provided, assign an available env_idx
        if env_idx is None:
            # [Lease-Reclaim] 回收超时占用的 env：被 SIGKILL 的采集 worker 持有的 env 不再发请求，
            # 超过 ENV_LEASE_TIMEOUT 未活动 → 视为泄漏，自动归还池（根治反复重启 collect 掏空池子）。
            now = time.monotonic()
            for held_idx in list(env_assigned_time.keys()):
                if held_idx not in free_env_index and now - env_assigned_time[held_idx] > ENV_LEASE_TIMEOUT:
                    free_env_index.add(held_idx)
                    logger.warning(f"[Lease-Reclaim] Environment {held_idx} idle {int(now - env_assigned_time[held_idx])}s > {ENV_LEASE_TIMEOUT}s, reclaimed (SIGKILL'd worker?)")
                    env_assigned_time.pop(held_idx, None)
            retry_count = 0
            while retry_count < MAX_RETRIES:
                if len(free_env_index) > 0:
                    env_idx = free_env_index.pop()
                    env_assigned_time[env_idx] = time.monotonic()
                    assigned = True
                    break
                retry_count += 1
                logger.info(f"[Retry {retry_count}/{MAX_RETRIES}] No available environment index, retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                # Reached max retries but still couldn't get env_idx
                logger.error("[Error] Reached max retries, unable to get available environment")
                return jsonify({'result': {'error': 'Unable to get available environment resource, please try again later'}})

        # [Lease-Renew] 带 env_idx 的活跃请求续约：更新活动时间，防止被 lease 超时误回收
        env_assigned_time[env_idx] = time.monotonic()
        # Call shop_agent function
        result = shop_agent(envs[env_idx], env_idx, action, idx, response)

        # If task is over, release the environment
        if 'over' in result and result['over']:
            free_env_index.add(env_idx)
            env_assigned_time.pop(env_idx, None)
            logger.info(f"[Task Over] Environment {env_idx} has been released")

    except Exception as e:
        # [Leak-Fix] 归还本次新分配却因异常未能正常使用的环境槽。
        # 原逻辑只在 result['over'] 时归还，shop_agent 抛异常会导致 idx 永久泄漏，
        # 累积后 free_env_index 被掏空，所有 reset 失败（"Unable to get available environment"）。
        if assigned and env_idx is not None and env_idx not in free_env_index:
            free_env_index.add(env_idx)
            env_assigned_time.pop(env_idx, None)
            logger.info(f"[Leak-Fix] Environment {env_idx} returned to free pool after exception")
        logger.exception(f"[Exception] Exception occurred while processing request: {str(e)}")
        return jsonify({'result': {'error': str(e)}})

    return jsonify({'result': result})


def initialize_environments() -> None:
    """
    Initialize all environments and add them to the free environment index.

    ``WebAgentTextEnv`` normally constructs a ``SimServer`` for every Gym
    environment.  A SimServer loads the complete product catalogue, goals and
    a LuceneSearcher, even though those resources are immutable during a
    rollout.  The only per-trajectory state lives in WebAgentTextEnv's browser
    and in SimServer.user_sessions (keyed by the environment index), so one
    server can safely back the entire pool.  This avoids loading the Lucene
    index once per slot and keeps the configured pool-size API contract.
    """
    global envs, free_env_index, env_max_num
    
    envs = []
    free_env_index = set()
    
    shared_server = None
    for i in range(env_max_num):
        logger.info(f"Environment {i} is being initialized")
        free_env_index.add(i)
        env_kwargs = dict(
            observation_mode='text',
            split="train",
            num_products=DEBUG_PROD_SIZE,
        )
        if shared_server is not None:
            env_kwargs['server'] = shared_server
        env = gym.make(
            'WebAgentTextEnv-v0',
            **env_kwargs,
        )
        if shared_server is None:
            shared_server = env.unwrapped.server
        envs.append(env)


if __name__ == '__main__':
    initialize_environments()
    app.run(host=SERVER_HOST, port=SERVER_PORT)
