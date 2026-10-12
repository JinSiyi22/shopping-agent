import logging
#logger.debug("准备 messages")
#logger.info("Agent 开始执行")
#logger.warning("即将达到最大步数")
#logger.error("Groq 请求失败")
#logger.critical("数据库完全不可用")

def configure_logging():

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )