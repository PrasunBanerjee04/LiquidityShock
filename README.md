# LiquidityShock
We implement a Stanford study on determining the best bid/ask following a liquidity shock (a large trade sweeping through several levels of the orderbook). We address the original paper's high-bias problem by designing and implementing a non-linear transformer in PyTorch to forecast whether or not the best bid/ask reverts to its original price or is significantly altered.
