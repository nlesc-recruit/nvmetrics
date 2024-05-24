import nvmetrics

if __name__ == "__main__":
    def print_metrics(metric_type, metrics):
        print("#" * 80)
        print(f"# {len(metrics)} {metric_type} metrics")
        print("#" * 80)
        print("\n".join(metrics))
        print()

    print_metrics("COUNTER", nvmetrics.queryMetrics(nvmetrics.NVPW_METRIC_TYPE_COUNTER))
    print_metrics("RATIO", nvmetrics.queryMetrics(nvmetrics.NVPW_METRIC_TYPE_RATIO))
    print_metrics("THROUGHPUT", nvmetrics.queryMetrics(nvmetrics.NVPW_METRIC_TYPE_THROUGHPUT))
