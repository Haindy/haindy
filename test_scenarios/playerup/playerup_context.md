target_type: mobile_adb

# Device selection
adb_serial: emulator-5554

# App launch target
app_package: com.playerup.mobile
app_activity: .MainActivity

# Notes
- MUST ASSUME that the android emulator is already running before test execution starts and the app is installed with the correct settings.
- We are only running in the test environment
- Follow the test plan IN ORDER, do not skip or merge test cases
- DO NOT extend beyond the scope of the test plan provided
- Assume the user data provided is valid and use it literally
- For email notifications the emulator has the gmail app signed in with the correct account. Assume it will work. 
