#!/usr/bin/env python3
"""
Simple OpenReview Conference Fetcher

A minimal script to login to OpenReview and get conference information for a single paper.

Requirements:
    pip install openreview-py

Usage:
    python simple_conference_fetcher.py
"""

import openreview
import sys

# dict of paper info
def get_conference_for_paper(paper_id: str, username: str = None, password: str = None):
    """
    Get conference information for a single paper ID.
    
    Args:
        paper_id: The OpenReview paper ID
        username: OpenReview username (optional for public papers)
        password: OpenReview password (optional for public papers)
    
    Returns:
        Dictionary with conference information
    """
    
    try:
        # Create OpenReview client
        if username and password:
            print("Logging in with credentials...")
            client = openreview.Client(
                baseurl='https://api.openreview.net',
                username=username,
                password=password
            )
            print("✓ Login successful!")
        else:
            print("Accessing public data (no login)...")
            client = openreview.Client(baseurl='https://api.openreview.net')
            print("✓ Connected to OpenReview API")
        
        # Get the paper
        print(f"Fetching paper: {paper_id}")
        note = client.get_note(paper_id)
        
        # Extract conference information
        invitation = note.invitation
        print(f"Raw invitation: {invitation}")
        
        # Parse conference from invitation
        conference_info = parse_conference_from_invitation(invitation)
        # import pdb; pdb.set_trace()
        
        # Get basic paper info
        result = {
            'paper_id': paper_id,
            'title': note.content.get('title', 'No title available'),
            'conference': conference_info['conference'],
            'year': conference_info['year'],
            'venue_full': conference_info['venue_full'],
            'raw_invitation': invitation,
            'authors': note.content.get('authors', []),
            'submission_date': note.cdate
        }
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_conference_from_invitation(invitation: str):
    """
    Parse conference information from OpenReview invitation string.
    
    Args:
        invitation: The invitation string from the paper
        
    Returns:
        Dictionary with parsed conference information
    """
    
    # Common patterns in OpenReview invitations:
    # ICLR.cc/2024/Conference/-/Submission
    # NeurIPS.cc/2023/Conference/-/Submission  
    # ICML.cc/2024/Conference/-/Submission
    # ACL.org/2024/Conference/-/Submission
    
    conference = "Unknown"
    year = "Unknown" 
    venue_full = invitation
    
    try:
        parts = invitation.split('/')
        if len(parts) > 0:
            venue_part = parts[0]
            
            # Handle .cc domains (ICLR, NeurIPS, ICML, etc.)
            if '.cc' in venue_part:
                domain_parts = venue_part.split('.')
                if len(domain_parts) >= 1:
                    conference = domain_parts[0].upper()
            
            # Handle .org domains (ACL, etc.)
            elif '.org' in venue_part:
                domain_parts = venue_part.split('.')
                if len(domain_parts) >= 1:
                    conference = domain_parts[0].upper()
            
            # Extract year (usually second part)
            if len(parts) > 1 and parts[1].isdigit():
                year = parts[1]
    
    except Exception as e:
        print(f"Warning: Could not parse invitation '{invitation}': {e}")
    
    return {
        'conference': conference,
        'year': year,
        'venue_full': venue_full
    }

# prettyprint for get_conference_for_paper
def get_paper_info(paper_id="SJD8YjCpW"):
    """Main function with example usage."""
    
    # Option 1: Try without login first (for public papers)
    print("=" * 50)
    print("SIMPLE OPENREVIEW CONFERENCE FETCHER")
    print("=" * 50)
    
    result = get_conference_for_paper(paper_id)
    
    if result:
        print("\n" + "=" * 30)
        print("RESULTS:")
        print("=" * 30)
        print(f"Paper ID: {result['paper_id']}")
        print(f"Title: {result['title']}")
        print(f"Conference: {result['conference']}")
        print(f"Year: {result['year']}")
        print(f"Full Venue: {result['venue_full']}")
        print(f"Authors: {', '.join(result['authors']) if result['authors'] else 'Not available'}")
        print(f"Submission Date: {result['submission_date']}")
    else:
        print("\n❌ Failed to fetch paper information")
        
        # If public access failed, try with credentials
        print("\nIf this is a private paper, you may need to login.")
        print("Uncomment the login section below and add your credentials.")

if __name__ == "__main__":
    get_paper_info()