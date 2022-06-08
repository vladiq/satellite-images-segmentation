import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
plt.axis("off")

from PIL import Image

palettes = {
    "av": [255, 255, 255]
    + [255, 255, 0]
    + [125, 125, 0]
    + [255, 150, 0]
    + [100, 150, 100],
    "ar": [255, 255, 255] + [30, 0, 0] + [255, 0, 0] + [50, 0, 0],
    "aw": [255, 255, 255] + [120, 120, 255] + [0, 0, 120] + [0, 0, 255],
    "ab": [255, 255, 255]
    + [0, 255, 0]
    + [130, 255, 200]
    + [0, 100, 0]
    + [100, 0, 100]
    + [128, 128, 128]
    + [0, 100, 255],
}


def main():
    ids = tqdm(range(3449))

    for idx in ids:
        images = dict()
        images["image"] = Image.open(f"../results/ab/{idx}_image.png")
        for category in ["ab", "ar", "av", "aw"]:
            for kind in ("pred", "gt"):
                images[f"{category}_{kind}"] = Image.fromarray(
                    np.array(
                        Image.open(f"../results/{category}/{idx}_mask_{kind}.png")
                    ),
                    mode="P",
                )
                images[f"{category}_{kind}"].putpalette(palettes[category])
                images[f"{category}_{kind}"] = images[f"{category}_{kind}"].convert(
                    "RGB"
                )

        fig, ax = plt.subplots(2, 5, figsize=(16, 9), constrained_layout=True)

        for row, kind in enumerate(("pred", "gt")):
            ax[row, 0].get_xaxis().set_visible(False)
            ax[row, 0].imshow(images["image"])
            ax[row, 0].set_ylabel(kind, rotation=0, fontsize=20)

            ax[row, 1].get_xaxis().set_visible(False)
            ax[row, 1].get_yaxis().set_visible(False)
            ax[row, 1].imshow(images[f"ab_{kind}"])
            ax[row, 1].set_title(f"ab")

            ax[row, 2].get_xaxis().set_visible(False)
            ax[row, 2].get_yaxis().set_visible(False)
            ax[row, 2].imshow(images[f"ar_{kind}"])
            ax[row, 2].set_title(f"ar")

            ax[row, 3].get_xaxis().set_visible(False)
            ax[row, 3].get_yaxis().set_visible(False)
            ax[row, 3].imshow(images[f"av_{kind}"])
            ax[row, 3].set_title(f"av")

            ax[row, 4].get_xaxis().set_visible(False)
            ax[row, 4].get_yaxis().set_visible(False)
            ax[row, 4].imshow(images[f"aw_{kind}"])
            ax[row, 4].set_title(f"aw")

        plt.savefig(f"../predictions_compiled/{idx}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
