# PlayerUp Sign-In Test Plan (Short)

## Test Plan Overview

**Document Version:** 1.0
**Date:** 2025-09-04
**Application:** PlayerUp Flutter Application
**Test Scope:** Sign-In and Session Management
**Test Type:** Manual Black Box Testing
**Test Environment Requirements:**
- Test and Production environments
- An iOS or Android device/simulator
- Network connectivity (WiFi and Cellular)
- Valid TeamSnap user accounts for testing
- Email accounts with inbox access for password reset

## Test Objectives

1. Validate all sign-in methods work correctly
2. Ensure proper error handling for invalid credentials
3. Verify session management and persistence
4. Test password reset functionality
5. Confirm TeamSnap OAuth integration
6. Validate security measures (account lockout, etc.)

---

## 1. Email Sign-In

### Test Case: TC-SI-001 - Successful Email Sign-In
**Priority:** Critical
**Type:** Functional
**Preconditions:** Existing account with email: qaagent@playerup.co, Password: Test1234!!

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Launch application | App opens |
| 2 | Navigate to Sign-In screen | Sign-In screen displayed |
| 3 | Enter email (qaagent@playerup.co) | Email accepted |
| 4 | Enter password (Test1234!!) | Password field populated |
| 5 | Click "Sign In" | Loading indicator displayed; authentication successful; user profile loaded; navigation to appropriate home screen based on user role (Player, Coach, or Admin) |

---

### Test Case: TC-SI-002 - Case-Insensitive Email Sign-In
**Priority:** Medium
**Type:** Functional
**Preconditions:** Account exists with email: qaagent@playerup.co

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Sign-In screen | Sign-In screen displayed |
| 2 | Enter email in different case (QAAGENT@PLAYERUP.CO) | Email accepted |
| 3 | Enter correct password (Test1234!!) | Password accepted |
| 4 | Click "Sign In" | Sign-in successful; home screen displayed |
| 5 | Tap the Settings icon in the bottom navigation bar | Settings screen displayed |
| 6 | Tap "Account Settings" | Account Settings screen displayed showing the account email address |
| 7 | Verify the displayed email matches qaagent@playerup.co | Email shown is qaagent@playerup.co confirming the correct account was accessed regardless of input case |
| 8 | Press back to return to Settings | Settings screen displayed |

---

### Test Case: TC-SI-003 - Incorrect Password
**Priority:** Critical
**Type:** Negative
**Preconditions:** Valid account exists

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Sign-In screen | Sign-In screen displayed |
| 2 | Enter valid email | Email accepted |
| 3 | Enter incorrect password | Password accepted in field |
| 4 | Click "Sign In" | Error message: "Incorrect username or password"; no specific indication if email exists; option to reset password visible |

---

### Test Case: TC-SI-004 - Non-Existent Email Account
**Priority:** High
**Type:** Negative
**Preconditions:** Email not registered: nonexistent@example.com

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to Sign-In screen | Sign-In screen displayed |
| 2 | Enter non-existent email | Email accepted |
| 3 | Enter any password | Password accepted |
| 4 | Click "Sign In" | Generic error: "Incorrect username or password"; does not reveal if account exists; sign-up option suggested |

---

## Test Execution Summary

| Test Case | Priority | Status |
|-----------|----------|--------|
| TC-SI-001 | Critical | Not Run |
| TC-SI-002 | Medium | Not Run |
| TC-SI-003 | Critical | Not Run |
| TC-SI-004 | High | Not Run |

## Notes

### Test Execution Guidelines
- Mark each step as completed when done
- Note Pass/Fail for each expected result
- Verify on multiple devices when possible

---

*End of Test Plan Document*
