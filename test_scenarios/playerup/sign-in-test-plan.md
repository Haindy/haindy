# PlayerUp Sign-In Test Plan

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

## 2. TeamSnap OAuth Sign-In

### Test Case: TC-SI-005 - TeamSnap OAuth Sign-In for Existing User
**Priority:** High
**Type:** Functional
**Preconditions:** Existing account created through TeamSnap: federico@fieldspace.co, ThiIsFine01!!

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "Continue with TeamSnap" | TeamSnap OAuth window opens |
| 2 | Enter TeamSnap credentials | Credentials accepted |
| 3 | Authorize PlayerUp | Authorization successful |
| 4 | Complete authentication | TeamSnap OAuth successful; teams from TeamSnap visible in app; navigation to home screen |

---

## 3. Password Reset

### Test Case: TC-SI-006 - Forgot Password Flow
**Priority:** Critical
**Type:** Functional
**Preconditions:** Existing account with email: qaagent@playerup.co, Password: Test1234!!

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Click "Forgot Password" on sign-in | Forgot password screen displayed |
| 2 | Enter email address | Email accepted |
| 3 | Submit request | Reset email sent confirmation |
| 4 | Check email for reset code | Reset email received within 60 seconds |
| 5 | Enter reset code | Reset code accepted |
| 6 | Enter new password (NewPass123!) | Password requirements enforced; new password accepted |
| 7 | Confirm new password | Password confirmed |
| 8 | Sign in with new password | Old password no longer works; successful sign-in with new password |
| 9 | Go to settings and change back the password to the original one: Test1234!! | Password will be changed back to the original one |

---

### Test Case: TC-SI-007 - Invalid Reset Code
**Priority:** Medium
**Type:** Negative
**Preconditions:** Password reset initiated

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Request password reset | Reset email sent |
| 2 | Enter incorrect code (000000) | Code entered |
| 3 | Submit code | Error: "Invalid verification code"; option to resend code; can retry with correct code |

---

## 4. Session Management

### Test Case: TC-SI-008 - Session Persistence
**Priority:** High
**Type:** Functional
**Preconditions:** User signed in successfully

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Sign in successfully | User authenticated |
| 2 | Close app completely (force quit) | App closed |
| 3 | Reopen app | User remains signed in; no re-authentication required; direct navigation to home screen; user profile and teams visible |

---

### Test Case: TC-SI-009 - Sign Out Functionality
**Priority:** High
**Type:** Functional
**Preconditions:** User signed in

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to settings/profile | Settings screen displayed |
| 2 | Click "Sign Out" | Confirmation dialog (if any) |
| 3 | Confirm sign out | User signed out successfully; navigation to sign-in screen; session cleared; no user data accessible |

---

## Test Execution Summary

| Test Case | Priority | Status |
|-----------|----------|--------|
| TC-SI-001 | Critical | Not Run |
| TC-SI-002 | Medium | Not Run |
| TC-SI-003 | Critical | Not Run |
| TC-SI-004 | High | Not Run |
| TC-SI-005 | High | Not Run |
| TC-SI-006 | Critical | Not Run |
| TC-SI-007 | Medium | Not Run |
| TC-SI-008 | High | Not Run |
| TC-SI-009 | High | Not Run |

## Notes

### Test Execution Guidelines
- Mark each step as completed when done
- Note Pass/Fail for each expected result
- Verify on multiple devices when possible

### Known Limitations
- Account lockout policies may vary by environment
- Session timeout duration configurable per environment
- TeamSnap OAuth requires valid TeamSnap test accounts

---

**Tester Name:** _______________________
**Test Date:** _______________________
**Environment:** Test / Production
**Platform:** iOS / Android
**Device Model:** _______________________
**OS Version:** _______________________

---

*End of Test Plan Document*
