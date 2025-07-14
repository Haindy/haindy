# Test Plan: Wikipedia Search and Article Verification

**Description**: Verify that users can search for topics on Wikipedia, navigate to the correct article, and confirm the article contains key sections, images, and infoboxes with a professional layout.
**Created**: 2025-07-14 21:05 UTC
**Requirements Source**: User Story #123: Wikipedia Search Functionality
**Tags**: wikipedia, search, articles
**Estimated Duration**: 10 minutes

## Summary
- **Total Test Cases**: 5
- **Total Test Steps**: 17
- **Priority Distribution**:
  - Critical: 1
  - High: 2
  - Medium: 2

## Test Cases

### TC001: Search for 'artificial intelligence' using Enter key

**Priority**: Critical
**Description**: Happy path: User searches for the topic 'artificial intelligence' on the Wikipedia home page and navigates to the correct article by pressing Enter.

#### Prerequisites
- Browser is installed
- User has internet connectivity

#### Test Steps
1. **Navigate to https://www.wikipedia.org/**
   - _Expected Result_: Wikipedia home page loads and displays the central search input with the placeholder 'Search Wikipedia'.
2. **Type "artificial intelligence" into the search input field in the center of the page**
   - _Expected Result_: The text "artificial intelligence" appears in the search input field.
   - _Depends on_: Step(s) 1
3. **Press the Enter key**
   - _Expected Result_: Browser navigates to the 'Artificial intelligence – Wikipedia' article page (URL contains '/wiki/Artificial_intelligence').
   - _Depends on_: Step(s) 2

#### Postconditions
- User is on the 'Artificial intelligence' article page

**Tags**: search, happy-path, enter-key

---

### TC002: Verify presence of key article sections

**Priority**: High
**Description**: Ensure that the 'Artificial intelligence' article contains the expected major sections: History, Applications, Ethics, etc.

#### Prerequisites
- User is on the 'Artificial intelligence' article page (TC001 postcondition)

#### Test Steps
1. **Locate the table of contents on the article page**
   - _Expected Result_: Table of contents is visible below the article title.
2. **Confirm that a section titled 'History' appears in the table of contents or as an H2 heading in the article body**
   - _Expected Result_: 'History' section heading is present.
   - _Depends on_: Step(s) 1
3. **Confirm that a section titled 'Applications' appears in the table of contents or as an H2 heading in the article body**
   - _Expected Result_: 'Applications' section heading is present.
   - _Depends on_: Step(s) 1
4. **Confirm that a section titled 'Ethics' appears in the table of contents or as an H2 heading in the article body**
   - _Expected Result_: 'Ethics' section heading is present.
   - _Depends on_: Step(s) 1

#### Postconditions
- Key sections are verified as present in the article

**Tags**: content, sections

---

### TC003: Verify presence of images and infobox

**Priority**: High
**Description**: Check that the 'Artificial intelligence' article includes at least one image in the content and an infobox on the right side.

#### Prerequisites
- User is on the 'Artificial intelligence' article page

#### Test Steps
1. **Scroll to the top-right area of the article page**
   - _Expected Result_: An infobox (a bordered box with summary information) is visible beside the article introduction.
2. **Verify that the infobox contains a title, image thumbnail, and at least one data field**
   - _Expected Result_: Infobox displays a heading (e.g., 'Artificial intelligence'), an image thumbnail, and summary fields such as 'Parent technology' or 'Developed in'.
   - _Depends on_: Step(s) 1
3. **Scroll down into the article content**
   - _Expected Result_: Main article content area is visible.
4. **Verify there is at least one image embedded within the article body**
   - _Expected Result_: At least one illustrative image (with caption) is visible within the article content.
   - _Depends on_: Step(s) 3

#### Postconditions
- Images and infobox are confirmed present on the article page

**Tags**: media, layout

---

### TC004: Search for a non-existent topic

**Priority**: Medium
**Description**: Negative scenario: Verify that searching for a gibberish term shows a search results page or a 'no results' message without errors.

#### Prerequisites
- User is on the Wikipedia home page

#### Test Steps
1. **Navigate to https://www.wikipedia.org/**
   - _Expected Result_: Wikipedia home page loads.
2. **Type "asdfghjklqwerty" into the search input field**
   - _Expected Result_: The text "asdfghjklqwerty" appears in the search input field.
   - _Depends on_: Step(s) 1
3. **Press the Enter key**
   - _Expected Result_: A search results page loads, indicating zero or a small number of results and optionally a 'The page "asdfghjklqwerty" does not exist' message.
   - _Depends on_: Step(s) 2

#### Postconditions
- Search gracefully handles non-existent topics without errors

**Tags**: search, negative

---

### TC005: Verify article professional layout and organization

**Priority**: Medium
**Description**: Validate that the 'Artificial intelligence' article is well-organized, with clear headings, consistent fonts, and a clean professional appearance.

#### Prerequisites
- User is on the 'Artificial intelligence' article page

#### Test Steps
1. **Observe the article title and heading hierarchy (H1, H2, H3)**
   - _Expected Result_: Title uses large, bold font; section headings use decreasing font sizes in a clear hierarchy.
2. **Verify consistent font style and spacing between paragraphs**
   - _Expected Result_: Paragraph text is legible, with uniform font and adequate line spacing.
3. **Check alignment of images, tables, and infobox against the text flow**
   - _Expected Result_: All images, tables, and the infobox align properly and do not overlap or cause layout breaks.

#### Postconditions
- Article layout meets professional quality standards

**Tags**: layout, usability

---
