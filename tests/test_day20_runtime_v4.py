from app.agents.multi_agent import route_after_research
from app.agents.shopping_agent import normalize_tool_arguments


def test_category_alias_ssd_maps_to_storage():
    result = normalize_tool_arguments(
        {
            "category": "ssd",
            "max_price": 50,
        }
    )
    assert result["category"] == "storage"
    assert result["max_price"] == 50


def test_category_alias_headphones_maps_to_headphone():
    result = normalize_tool_arguments(
        {
            "category": "headphones",
        }
    )
    assert result["category"] == "headphone"


def test_research_route_uses_workers_for_semantic_rerank():
    assert (
        route_after_research(
            {
                "research_tool": "rerank_search_products",
            }
        )
        == "parallel_research"
    )


def test_research_route_skips_workers_for_structured_search():
    assert (
        route_after_research(
            {
                "research_tool": "search_products",
            }
        )
        == "recommendation"
    )


def test_research_route_skips_workers_for_detail():
    assert (
        route_after_research(
            {
                "research_tool": "get_product_detail",
            }
        )
        == "recommendation"
    )
