# 文件用途：必胜策略模块
# 最后修改：2025-12-04
# 主要功能：根据玩家手势返回能战胜的机器人手势
# 使用说明：入口脚本统一调用以保持策略一致。

def get_winning_move(user_move: str) -> str:
    """根据玩家手势返回机器人必胜手势；未知或未识别时返回 'waiting'"""
    mapping = {
        'rock': 'paper',
        'paper': 'scissors',
        'scissors': 'rock',
    }
    return mapping.get(user_move, 'waiting')
