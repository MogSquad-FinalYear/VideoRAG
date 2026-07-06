"""
VideoRAG — Agent Tool Definitions
Tool schemas for Groq function calling.
Includes all tools for the full architecture: caption, visual, speech, OCR, detection, scene graph.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_caption",
            "description": "Search for video frames by their scene descriptions. Captions are short and generic (e.g., 'a man walking on a road'). Best for general scene/activity queries, NOT for identifying specific people or characters. Returns matching frames with captions and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what to search for (e.g., 'person holding a bag', 'red car on the street')"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within. Omit to search all videos."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10, use higher for exhaustive search)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_visual_similarity",
            "description": "Search for visually similar frames using CLIP embeddings. BEST tool for finding specific people, characters, objects, logos, or visual appearances. More accurate than caption search for identifying WHO or WHAT is in a frame. ALWAYS use this for questions about specific people or characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Detailed visual description of what to find (e.g., 'Captain America in his suit', 'a man in a blue uniform with a shield')"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10, use higher for exhaustive search)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": "Search through speech transcripts of videos. Use this when the user asks about something that was SAID or SPOKEN in the video, or when looking for specific words, phrases, or dialog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in speech transcripts"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_image",
            "description": "Search for frames visually similar to a reference image. Use when a user provides a reference image and wants to find similar-looking frames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "File path to the reference image to search for"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_clip_info",
            "description": "Get information about a specific time range in a video, including the frames in that range. Use this to retrieve details about a specific moment identified by other search tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The video ID"
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds"
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds"
                    }
                },
                "required": ["video_id", "start_time", "end_time"]
            }
        }
    },
    # ── Phase 3: OCR Tool ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Search for on-screen text (OCR) found in video frames. Use this when the user asks about text visible in the video — signs, labels, titles, subtitles, license plates, book covers, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in OCR-extracted content"
                    },
                    "video_id": {
                        "type": "string",
                        "description": "Optional: specific video ID to search within."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    # ── Phase 4: Detection Tools ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "count_objects",
            "description": "Count how many instances of a specific object class appear in video frames. Use for 'how many cars/people/books are there?' type questions. Returns the maximum count seen in any single frame and the number of frames where the object appears.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Object class to count (e.g., 'car', 'person', 'book', 'chair'). Use COCO class names."
                    },
                    "video_id": {
                        "type": "string",
                        "description": "The video ID to search within."
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Optional: start of time range in seconds"
                    },
                    "end_time": {
                        "type": "number",
                        "description": "Optional: end of time range in seconds"
                    }
                },
                "required": ["class_name", "video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "locate_object",
            "description": "Find the spatial location of a specific object in video frames. Returns bounding box coordinates and a spatial description (e.g., 'top-left', 'center', 'bottom-right'). Use for 'where is the X in the video?' type questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Object class to locate (e.g., 'car', 'person', 'book'). Use COCO class names."
                    },
                    "video_id": {
                        "type": "string",
                        "description": "The video ID to search within."
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Optional: start of time range in seconds"
                    },
                    "end_time": {
                        "type": "number",
                        "description": "Optional: end of time range in seconds"
                    }
                },
                "required": ["class_name", "video_id"]
            }
        }
    },
    # ── Phase 5: Scene Graph Tool ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_relations",
            "description": "Get spatial relationships between objects in video frames (e.g., 'book on top of table', 'person left of car', 'cat near sofa'). Use for questions about how objects relate to each other spatially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The video ID to search within."
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Optional: start of time range in seconds"
                    },
                    "end_time": {
                        "type": "number",
                        "description": "Optional: end of time range in seconds"
                    }
                },
                "required": ["video_id"]
            }
        }
    },
]
