import logging


def configure_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        force=True,
    )

    # 我们自己的业务日志保留 INFO
    logging.getLogger(
        "app"
    ).setLevel(
        logging.INFO
    )

    # 第三方库降噪
    noisy_loggers = [
        "httpx",
        "httpx2",
        "openai",
        "openai._base_client",
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "urllib3",
        "jieba",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(
            logger_name
        ).setLevel(
            logging.WARNING
        )