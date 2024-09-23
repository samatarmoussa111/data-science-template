import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 1]

plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

plt.style.use('default')
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100


# -------- ------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").sort_values("category").reset_index(drop=True)

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index(drop=True)

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"

all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("accelerometer")
            ax.set_xlabel("Samples")
            plt.title(f"{label} - {participant}")
            plt.legend()
            
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyrescope")
            ax.set_xlabel("Samples")
            plt.title(f"{label} - {participant}")
            plt.legend()
      


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"

compined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
compined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
compined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
ax[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[0].set_ylabel("accelerometer")
ax[1].set_ylabel("Gyrescope")
ax[1].set_xlabel("Samples")




# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()
for label in labels:
    for participant in participants:
        compined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop=True)
        if len(compined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            compined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            compined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            ax[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
            ax[0].set_ylabel("accelerometer")
            ax[1].set_ylabel("Gyrescope")
            ax[1].set_xlabel("Samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
