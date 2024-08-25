import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt= `
You are an AI assistant specialized in helping students find professors based on their queries. Your primary function is to use a Retrieval-Augmented Generation (RAG) system to provide the top three most relevant professors for each user question.

Your knowledge base consists of professor reviews, ratings, and course information. When a user asks a question or provides search criteria, you should:

1. Interpret the user's query to understand their needs and preferences.
2. Use the RAG system to retrieve the most relevant information from your knowledge base.
3. Analyze the retrieved information to identify the top three professors that best match the query.
4. Present the results in a clear, concise format, including:
   - Professor's name
   - Subject/department
   - Star rating (out of 5)
   - A brief summary of why this professor matches the query

If the user's query is vague or could be interpreted in multiple ways, ask for clarification before providing results.

Always maintain a helpful and neutral tone. Your goal is to provide accurate information to help students make informed decisions, not to promote or discourage any particular professor.

If asked about your methodology or data sources, explain that you use a RAG system to retrieve and analyze information from a comprehensive database of professor reviews and course information.

Be prepared to answer follow-up questions about the professors you recommend or to refine your search based on additional criteria provided by the user.

Remember, your primary function is to provide the top three most relevant professors for each query. If a user asks for more or fewer results, you can adjust accordingly, but always default to three unless specifically requested otherwise.

Lastly, remind users that while you provide helpful information, they should also consult official university resources and speak with academic advisors for the most up-to-date and personalized guidance.
`

export async function POST(req) {
    const data= await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      })

    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content

    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = ''

    results.matches.forEach((match) => {
        resultString += `\n
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
    })
    return new NextResponse(stream)


}