target_type: mobile_adb

# Device selection
adb_serial: emulator-5554

# App launch target
app_package: com.playerup.mobile
app_activity: .MainActivity

# Optional fallback path (model may use these if needed)
adb_commands:
  - adb devices
  - adb -s emulator-5554 shell am start -n com.playerup.mobile/.MainActivity

# Notes
- Android emulator is already running before test execution starts.
- Keep the app in portrait mode.
