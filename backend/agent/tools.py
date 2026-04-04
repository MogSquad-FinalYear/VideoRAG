"""
VideoRAG — Agent Tool Definitions
Tool schemas for Groq function calling.
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
    }
]
