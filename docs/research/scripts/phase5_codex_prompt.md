You are an automated agent working on an OSWorld benchmark task. The desktop you control is an Ubuntu 22.04 GNOME guest running inside a VM. You drive that desktop through the `vm-haindy` skill — there is no other way for you to interact with the VM.

# The task you must complete

I am currently using an Ubuntu system, and I have wrongly deleted a poster of party night. Could you help me recover it from the Trash?

# Rules

1. Use the `vm-haindy` skill EXCLUSIVELY for any interaction with the VM. The skill is fully documented; follow its session lifecycle (new → act/explore → close).
2. Do NOT run shell commands directly on the host. Do NOT use shell to manipulate files in the VM. The task must be completed through the GUI exactly as a real user would — opening Files (Nautilus), navigating to Trash, restoring the file.
3. Do NOT verify your work via shell, filesystem checks, or any side-channel. An external evaluator will check the result independently.
4. Be efficient. The trash recovery should take roughly 5 to 10 GUI actions. If you exceed 20 actions or 8 minutes of wall-clock without progress, stop and report what you did.
5. When you believe the task is complete (you have restored the file from Trash to the Desktop), close the vm-haindy session and exit. Do not run any more commands.

# Reporting

When you stop, report briefly:
- The vm-haindy session_id you used
- The number of actions you took
- Whether you believe the file was restored successfully
