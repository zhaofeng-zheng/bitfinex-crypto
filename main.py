from Exchange import MyExchange
from optimizer.optimizer import Optimizer


optimizer = Optimizer(symbol='ETH/USD',
                      recentPerformanceTimeDuration=5)
if __name__ == "__main__":
    optimizer.screen_for_best_parameter()
