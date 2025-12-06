#!/usr/bin/env python3
"""
Simple ICLR Data Fetcher - Clean Implementation

A straightforward script to fetch ICLR papers and reviews with proper pagination.
No overengineering, just working code.
"""

import openreview
import json
import time
from typing import List, Dict
from tqdm import tqdm

def get_all_notes(client, **kwargs):
    """Simple pagination wrapper that always returns a list."""
    all_notes = []
    offset = 0
    limit = 1000
    
    while True:
        try:
            kwargs_with_pagination = kwargs.copy()
            kwargs_with_pagination['offset'] = offset
            kwargs_with_pagination['limit'] = limit
            
            notes = client.get_notes(**kwargs_with_pagination)
            
            # Handle the case where notes might be None
            if not notes:  # This handles None, [], and other falsy values
                break
                
            all_notes.extend(notes)
            
            if len(notes) < limit:
                break
                
            offset += limit
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break
    
    return all_notes

def get_papers_for_year(client, year):
    """Get all papers for a specific ICLR year."""
    print(f"Fetching ICLR {year} papers...")
    
    # Try the main pattern first
    invitation = f'ICLR.cc/{year}/Conference/-/Blind_Submission'
    papers = get_all_notes(client, invitation=invitation)
    
    if not papers:
        # Try alternative patterns
        alternatives = [
            f'ICLR.cc/{year}/Conference/-/Submission',
            f'ICLR.cc/{year}/-/Submission'
        ]
        for alt in alternatives:
            papers = get_all_notes(client, invitation=alt)
            if papers:
                break
    
    # Process papers
    processed_papers = []
    for paper in papers:
        processed_papers.append({
            'paper_id': paper.id,
            'title': getattr(paper.content, 'title', 'No title') if hasattr(paper.content, 'title') else paper.content.get('title', 'No title'),
            'abstract': getattr(paper.content, 'abstract', '') if hasattr(paper.content, 'abstract') else paper.content.get('abstract', ''),
            'authors': getattr(paper.content, 'authors', []) if hasattr(paper.content, 'authors') else paper.content.get('authors', []),
            'year': year,
            'venue': f'ICLR {year}'
        })
    
    print(f"Found {len(processed_papers)} papers for ICLR {year}")
    return processed_papers

def get_reviews_for_paper(client, paper_id):
    """Get reviews for a specific paper."""
    reviews = []
    
    try:
        # Get all forum notes (includes reviews)
        forum_notes = get_all_notes(client, forum=paper_id)
        
        for note in forum_notes:
            # Check if this looks like a review
            if 'review' in note.invitation.lower():
                reviews.append({
                    'paper_id': paper_id,
                    'review_id': note.id,
                    'rating': note.content.get('rating', 'Not specified'),
                    'confidence': note.content.get('confidence', 'Not specified'),
                    'review': note.content.get('review', 'No review text'),
                    'summary': note.content.get('summary', 'No summary')
                })
    except Exception as e:
        print(f"Error getting reviews for {paper_id}: {e}")
    
    return reviews

def main():
    """Main function."""
    print("ICLR 2018-2022 Data Fetcher")
    print("=" * 40)
    
    # Create client
    client = openreview.Client(baseurl='https://api.openreview.net')
    
    years = ['2018', '2019', '2020', '2021', '2022']
    all_data = {}
    
    for year in years:
        print(f"\nProcessing ICLR {year}...")
        
        # Get papers
        papers = get_papers_for_year(client, year)
        
        # Get reviews for each paper
        all_reviews = []
        for paper in tqdm(papers, desc=f"Reviews {year}"):  # Limit to first 100 for testing
            reviews = get_reviews_for_paper(client, paper['paper_id'])
            all_reviews.append(reviews)
            time.sleep(0.1)  # Rate limiting

        all_data[year] = {
            'papers': papers,
            'reviews': all_reviews
        }

        # Save year data
        with open(f'/share/garg/openreview_data/iclr_{year}_simple.json', 'w') as f:
            json.dump(all_data[year], f, indent=2, default=str)
        
        print(f"ICLR {year}: {len(papers)} papers, {len(all_reviews)} reviews")
    
    print("\nDone!")

if __name__ == "__main__":
    main()