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
- Phone number with SMS capability

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

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Launch application | App opens | [ ] |
| [ ] | 2. Navigate to Sign-In screen | Sign-In screen displayed | [ ] |
| [ ] | 3. Enter email (qaagent@playerup.co) | Email accepted | [ ] |
| [ ] | 4. Enter password (Test1234!!) | Password field populated | [ ] |
| [ ] | 5. Click "Sign In" | • Loading indicator displayed<br>• Authentication successful<br>• User profile information loaded<br>• Navigation based on user role:<br>&nbsp;&nbsp;- Player → Player home screen<br>&nbsp;&nbsp;- Coach → Coach home screen<br>&nbsp;&nbsp;- Admin → Admin organization home screen | [ ] |

---

### Test Case: TC-SI-002 - Case-Insensitive Email Sign-In
**Priority:** Medium  
**Type:** Functional  
**Preconditions:** Account exists with email: qaagent@playerup.co

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In screen | Sign-In screen displayed | [ ] |
| [ ] | 2. Enter email in different case (qaagent@playerup.co) | Email accepted | [ ] |
| [ ] | 3. Enter correct password | Password accepted | [ ] |
| [ ] | 4. Click "Sign In" | • Sign-in successful regardless of email case<br>• Correct user account accessed | [ ] |

---

### Test Case: TC-SI-003 - Incorrect Password
**Priority:** Critical  
**Type:** Negative  
**Preconditions:** Valid account exists

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In screen | Sign-In screen displayed | [ ] |
| [ ] | 2. Enter valid email | Email accepted | [ ] |
| [ ] | 3. Enter incorrect password | Password accepted in field | [ ] |
| [ ] | 4. Click "Sign In" | • Error message: "Incorrect username or password"<br>• No specific indication if email exists<br>• Option to reset password visible | [ ] |

---

### Test Case: TC-SI-004 - Non-Existent Email Account
**Priority:** High  
**Type:** Negative  
**Preconditions:** Email not registered: nonexistent@example.com

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In screen | Sign-In screen displayed | [ ] |
| [ ] | 2. Enter non-existent email | Email accepted | [ ] |
| [ ] | 3. Enter any password | Password accepted | [ ] |
| [ ] | 4. Click "Sign In" | • Generic error: "Incorrect username or password"<br>• Does not reveal if account exists<br>• Sign-up option suggested | [ ] |

---

### Test Case: TC-SI-005 - Multiple Failed Login Attempts
**Priority:** High  
**Type:** Security  
**Preconditions:** Valid account exists

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In screen | Sign-In screen displayed | [ ] |
| [ ] | 2. Enter valid email | Email accepted | [ ] |
| [ ] | 3. Enter wrong password and submit | First attempt fails | [ ] |
| [ ] | 4. Repeat step 3 two more times | Second and third attempts fail | [ ] |
| [ ] | 5. Attempt fourth login | Warning about account lockout displayed | [ ] |
| [ ] | 6. Continue failed attempts (5+ times) | • Account temporarily locked<br>• Message about waiting period or contacting support<br>• Password reset option highlighted | [ ] |

---

## 2. Phone Number Sign-In

### Test Case: TC-SI-006 - Successful Phone Number Sign-In
**Priority:** High  
**Type:** Functional  
**Preconditions:** Account with phone: 555-123-4567

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In | Sign-In screen displayed | [ ] |
| [ ] | 2. Toggle to phone input | Input changes to phone format | [ ] |
| [ ] | 3. Enter phone number (555-123-4567) | Phone number formatted correctly | [ ] |
| [ ] | 4. Enter password | Password accepted | [ ] |
| [ ] | 5. Click "Sign In" | • Authentication successful<br>• Navigation to appropriate home screen | [ ] |

---

### Test Case: TC-SI-007 - Invalid Phone Format
**Priority:** Medium  
**Type:** Negative  
**Preconditions:** None

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In | Sign-In screen displayed | [ ] |
| [ ] | 2. Toggle to phone input | Input changes to phone format | [ ] |
| [ ] | 3. Enter invalid phone (123) | Validation error shown | [ ] |
| [ ] | 4. Enter letters (abcdefgh) | Input rejected or filtered | [ ] |
| [ ] | 5. Try to sign in | • Cannot proceed with invalid format<br>• Clear error message about phone format | [ ] |

---

## 3. TeamSnap OAuth Sign-In

### Test Case: TC-SI-008 - TeamSnap OAuth Sign-In for Existing User
**Priority:** High  
**Type:** Functional  
**Preconditions:** Existing account created through TeamSnap

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Click "Continue with TeamSnap" | TeamSnap OAuth window opens | [ ] |
| [ ] | 2. Enter TeamSnap credentials | Credentials accepted | [ ] |
| [ ] | 3. Authorize PlayerUp | Authorization successful | [ ] |
| [ ] | 4. Complete authentication | • TeamSnap OAuth successful<br>• Teams from TeamSnap visible in app<br>• Navigation to home screen | [ ] |

---

### Test Case: TC-SI-009 - TeamSnap OAuth - New User
**Priority:** High  
**Type:** Functional  
**Preconditions:** TeamSnap account not linked to PlayerUp

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Click "Continue with TeamSnap" | TeamSnap OAuth window opens | [ ] |
| [ ] | 2. Enter TeamSnap credentials | Credentials accepted | [ ] |
| [ ] | 3. Authorize PlayerUp | Authorization successful | [ ] |
| [ ] | 4. Return to app | • New account creation flow triggered<br>• Profile setup required<br>• Teams imported from TeamSnap | [ ] |

---

## 4. Password Reset

### Test Case: TC-SI-010 - Forgot Password Flow
**Priority:** Critical  
**Type:** Functional  
**Preconditions:** Existing account with known email

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Click "Forgot Password" on sign-in | Forgot password screen displayed | [ ] |
| [ ] | 2. Enter email address | Email accepted | [ ] |
| [ ] | 3. Submit request | Reset email sent confirmation | [ ] |
| [ ] | 4. Check email for reset code | Reset email received within 60 seconds | [ ] |
| [ ] | 5. Enter reset code | Reset code accepted | [ ] |
| [ ] | 6. Enter new password (NewPass123!) | • Password requirements enforced<br>• New password accepted | [ ] |
| [ ] | 7. Confirm new password | Password confirmed | [ ] |
| [ ] | 8. Sign in with new password | • Old password no longer works<br>• Successful sign-in with new password | [ ] |

---

### Test Case: TC-SI-011 - Invalid Reset Code
**Priority:** Medium  
**Type:** Negative  
**Preconditions:** Password reset initiated

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Request password reset | Reset email sent | [ ] |
| [ ] | 2. Enter incorrect code (000000) | Code entered | [ ] |
| [ ] | 3. Submit code | • Error: "Invalid verification code"<br>• Option to resend code<br>• Can retry with correct code | [ ] |

---

## 5. Session Management

### Test Case: TC-SI-012 - Session Persistence
**Priority:** High  
**Type:** Functional  
**Preconditions:** User signed in successfully

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Sign in successfully | User authenticated | [ ] |
| [ ] | 2. Close app completely (force quit) | App closed | [ ] |
| [ ] | 3. Reopen app | • User remains signed in<br>• No re-authentication required<br>• Direct navigation to home screen<br>• User profile and teams visible | [ ] |

---

### Test Case: TC-SI-013 - Session Timeout
**Priority:** Medium  
**Type:** Functional  
**Preconditions:** User signed in

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Sign in successfully | User authenticated | [ ] |
| [ ] | 2. Leave app idle for extended period | App in background/idle | [ ] |
| [ ] | 3. Return to app after timeout | • Session expired message (if applicable)<br>• Redirect to sign-in screen<br>• Previous email pre-filled for convenience | [ ] |

---

### Test Case: TC-SI-014 - Sign Out Functionality
**Priority:** High  
**Type:** Functional  
**Preconditions:** User signed in

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to settings/profile | Settings screen displayed | [ ] |
| [ ] | 2. Click "Sign Out" | Confirmation dialog (if any) | [ ] |
| [ ] | 3. Confirm sign out | • User signed out successfully<br>• Navigation to sign-in screen<br>• Session cleared<br>• No user data accessible | [ ] |

---

## 6. Edge Cases

### Test Case: TC-SI-015 - Sign-In with No Network
**Priority:** High  
**Type:** Negative  
**Preconditions:** None

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Disable network connectivity | Network disabled | [ ] |
| [ ] | 2. Attempt sign-in | Sign-in screen accessible | [ ] |
| [ ] | 3. Enter credentials and submit | • Network error detected immediately<br>• Clear error message: "No internet connection"<br>• Retry option provided<br>• Credentials preserved for retry | [ ] |

---

### Test Case: TC-SI-016 - Remember Me Functionality
**Priority:** Low  
**Type:** Functional  
**Preconditions:** Remember Me option exists

| Executed | Test Steps | Expected Results | Pass/Fail |
|----------|------------|------------------|-----------|  
| [ ] | 1. Navigate to Sign-In | Sign-in screen displayed | [ ] |
| [ ] | 2. Enter credentials | Credentials entered | [ ] |
| [ ] | 3. Check "Remember Me" (if available) | Option selected | [ ] |
| [ ] | 4. Sign in successfully | User authenticated | [ ] |
| [ ] | 5. Sign out | User signed out | [ ] |
| [ ] | 6. Return to sign-in screen | • Email pre-filled<br>• Only password required<br>• Remember Me still checked | [ ] |

---

## Test Execution Summary

| Test Case | Priority | Status | Pass/Fail |
|-----------|----------|--------|-----------|
| TC-SI-001 | Critical | [ ] Not Run | [ ] |
| TC-SI-002 | Medium | [ ] Not Run | [ ] |
| TC-SI-003 | Critical | [ ] Not Run | [ ] |
| TC-SI-004 | High | [ ] Not Run | [ ] |
| TC-SI-005 | High | [ ] Not Run | [ ] |
| TC-SI-006 | High | [ ] Not Run | [ ] |
| TC-SI-007 | Medium | [ ] Not Run | [ ] |
| TC-SI-008 | High | [ ] Not Run | [ ] |
| TC-SI-009 | High | [ ] Not Run | [ ] |
| TC-SI-010 | Critical | [ ] Not Run | [ ] |
| TC-SI-011 | Medium | [ ] Not Run | [ ] |
| TC-SI-012 | High | [ ] Not Run | [ ] |
| TC-SI-013 | Medium | [ ] Not Run | [ ] |
| TC-SI-014 | High | [ ] Not Run | [ ] |
| TC-SI-015 | High | [ ] Not Run | [ ] |
| TC-SI-016 | Low | [ ] Not Run | [ ] |

## Notes

### Test Execution Guidelines
- Check each step as completed ([ ] → [x])
- Mark Pass/Fail for each expected result
- Test both email and phone variations where applicable
- Verify on multiple devices when possible

### Known Limitations
- Account lockout policies may vary by environment
- Session timeout duration configurable per environment
- TeamSnap OAuth requires valid TeamSnap test accounts

---

**Tester Name:** _______________________  
**Test Date:** _______________________  
**Environment:** [ ] Test [ ] Production  
**Platform:** [ ] iOS [ ] Android  
**Device Model:** _______________________  
**OS Version:** _______________________

---

*End of Test Plan Document*