from pathlib import Path

import yaml

from reporter import Reporter

if __name__ == "__main__":
    with open(Path("config.yml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    reporter = Reporter(config=config)
    reporter.test_module(bilinear_type="bilinear", module="UNet")
    reporter.test_module(bilinear_type="bilinear", module="UNetAvg")
    reporter.test_module(bilinear_type="bilinear", module="UNetLeakyAvg")
    reporter.test_module(bilinear_type="bilinear", module="NestedUNet")
    reporter.test_module(bilinear_type="bilinear", module="NestedUNetAvg")
    reporter.test_module(bilinear_type="bilinear", module="NestedUNetLeakyAvg")
    reporter.test_module(bilinear_type="transposed", module="NestedUNet")

    reporter.dump_all_models_stats()
