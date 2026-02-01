from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("charts/", views.charts, name="charts"),
    path("table/", views.table, name="table"),

    # JSON APIs
    path("api/filters/", views.api_filters, name="api_filters"),
    path("api/summary/", views.api_summary, name="api_summary"),
    path("api/top-cities/", views.api_top_cities, name="api_top_cities"),
    path("api/top-states/", views.api_top_states, name="api_top_states"),
    path("api/price-buckets/", views.api_price_buckets, name="api_price_buckets"),
    path("api/price-hist/", views.api_price_hist, name="api_price_hist"),
    path("api/scatter-rating-price/", views.api_scatter_rating_price, name="api_scatter_rating_price"),

    # ✅ Mini rows for overview quick preview
    path("api/mini-rows/", views.api_mini_rows, name="api_mini_rows"),
]
