import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'

export const Trading: React.FC = () => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Trading Activity</CardTitle>
          <CardDescription>Orders, trades, and execution history</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <p className="text-gray-400 mb-2">Trading history coming soon</p>
            <p className="text-sm text-gray-500">View your orders and trade execution history</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}